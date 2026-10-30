from contextlib import ExitStack
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import rasterio
from rasterio.enums import ColorInterp
from rasterio.transform import from_origin
import torch

from infer_geotiff import run_inference


class InputValidityTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.source = self.root / "input.tif"
        self.output = self.root / "outputs"
        self.args = SimpleNamespace(
            input=str(self.source), output_dir=str(self.output), name="result",
            overwrite=False, tile_size=4, stride=3, batch_size=2, threshold=0.5,
            save_prob=True, rgb_bands=[1, 2, 3], mean=[0, 0, 0], std=[1, 1, 1],
            upsample_to_gsd=None, polygonize=False,
        )
        cpu = torch.device("cpu")
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch("infer_geotiff.torch.cuda.is_available", return_value=True))
        stack.enter_context(patch("infer_geotiff.torch.device", return_value=cpu))
        self.load_model = stack.enter_context(patch("infer_geotiff.load_model"))
        self.predict = stack.enter_context(patch(
            "infer_geotiff.predict_batch",
            side_effect=lambda model, batch: torch.full(batch.shape[:1] + batch.shape[-2:], 0.75),
        ))

    def write_source(self, rgb=None, nodata=None, mask=None, alpha=False):
        if rgb is None:
            rgb = np.full((3, 8, 9), 128, dtype=np.uint8)
        with rasterio.open(self.source, "w", driver="GTiff", width=9, height=8,
                           count=rgb.shape[0], dtype=rgb.dtype, nodata=nodata,
                           crs="EPSG:3857", transform=from_origin(100, 100, 1, 1)) as dst:
            dst.write(rgb)
            if mask is not None:
                dst.write_mask(mask)
            if alpha:
                dst.colorinterp = (ColorInterp.red, ColorInterp.green, ColorInterp.blue, ColorInterp.alpha)

    def assert_no_outputs(self):
        self.assertEqual(list(self.output.glob("*")), [])

    def test_nodata_rejected_before_resampling_and_model_loading(self):
        rgb = np.full((3, 8, 9), 128, dtype=np.uint8)
        rgb[:, :2, :3] = 0
        self.write_source(rgb, nodata=0)
        for target in [None, 0.5]:
            with self.subTest(target=target):
                self.args.upsample_to_gsd = target
                with patch("infer_geotiff.maybe_upsampled_source") as resample:
                    with self.assertRaisesRegex(ValueError, "NoData or masked pixels"):
                        run_inference(self.args)
                    resample.assert_not_called()
                self.load_model.assert_not_called()
                self.predict.assert_not_called()
                self.assert_no_outputs()

    def test_dataset_mask_and_alpha_invalidity_rejected(self):
        for kind in ["mask", "alpha"]:
            with self.subTest(kind=kind):
                valid = np.full((8, 9), 255, dtype=np.uint8)
                valid[:2, :3] = 0
                if kind == "mask":
                    self.write_source(mask=valid)
                else:
                    self.source.unlink()
                    self.source.with_suffix(".tif.msk").unlink(missing_ok=True)
                    rgb = np.full((4, 8, 9), 128, dtype=np.uint8)
                    rgb[3] = valid
                    self.write_source(rgb, alpha=True)
                with self.assertRaisesRegex(ValueError, "NoData or masked pixels"):
                    run_inference(self.args)
                self.load_model.assert_not_called()
                self.assert_no_outputs()

    def test_nonfinite_and_out_of_range_input_rejected(self):
        for value in [np.nan, np.inf, -np.inf, -1, 256]:
            with self.subTest(value=value):
                rgb = np.full((3, 8, 9), 128, dtype=np.float32)
                rgb[0, 7, 8] = value
                self.write_source(rgb)
                with self.assertRaises(ValueError):
                    run_inference(self.args)
                self.load_model.assert_not_called()
                self.assert_no_outputs()

    def test_nonfinite_predictions_rejected_without_output(self):
        self.write_source()
        for value in [np.nan, np.inf, -np.inf]:
            with self.subTest(value=value):
                self.predict.side_effect = lambda model, batch: torch.full(
                    batch.shape[:1] + batch.shape[-2:], value)
                with self.assertRaisesRegex(RuntimeError, "Model predictions contain NaN or Inf"):
                    run_inference(self.args)
                self.assert_no_outputs()

    def test_nonfinite_merged_probabilities_rejected_without_output(self):
        self.write_source()
        with patch("infer_geotiff.blend_weight", return_value=np.full((4, 4), np.inf, dtype=np.float32)), \
             np.errstate(invalid="ignore"):
            with self.assertRaisesRegex(RuntimeError, "Merged probabilities contain NaN or Inf"):
                run_inference(self.args)
        self.assert_no_outputs()

    def test_valid_zero_pixels_still_write_valid_predictions(self):
        self.write_source(np.zeros((3, 8, 9), dtype=np.uint8))
        run_inference(self.args)
        with rasterio.open(self.output / "result_mask.tif") as mask, \
             rasterio.open(self.output / "result_prob.tif") as prob:
            self.assertIsNone(mask.nodata)
            np.testing.assert_array_equal(mask.read_masks(1), np.full((8, 9), 255, dtype=np.uint8))
            np.testing.assert_array_equal(mask.read(1), np.ones((8, 9), dtype=np.uint8))
            np.testing.assert_allclose(prob.read(1), 0.75)


if __name__ == "__main__":
    unittest.main()
