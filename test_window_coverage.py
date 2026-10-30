import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import rasterio
import torch
from rasterio.transform import from_origin

from infer_geotiff import compute_starts, run_inference


class WindowCoverageTest(unittest.TestCase):
    def test_valid_windows_cover_every_pixel(self):
        for length, tile, stride in [(2048, 512, 384), (2048, 512, 512),
                                     (2051, 512, 384), (127, 512, 384), (1, 1, 1)]:
            with self.subTest(length=length, tile=tile, stride=stride):
                starts = compute_starts(length, tile, stride)
                covered = np.zeros(length, dtype=bool)
                for start in starts:
                    covered[start:start + tile] = True
                self.assertTrue(covered.all())
                self.assertEqual(starts[0], 0)
                self.assertEqual(starts[-1], max(0, length - tile))

    def test_invalid_windows_are_rejected_even_for_small_images(self):
        for length, tile, stride in [(2048, 512, 768), (127, 512, 768),
                                     (2048, 0, 384), (2048, -1, 384),
                                     (2048, 512, 0), (2048, 512, -1),
                                     (0, 512, 384), (-1, 512, 384)]:
            with self.subTest(length=length, tile=tile, stride=stride):
                with self.assertRaises(ValueError):
                    compute_starts(length, tile, stride)

    def test_invalid_cli_settings_fail_before_model_loading(self):
        for tile, stride, batch in [(512, 768, 1), (0, 1, 1), (512, 0, 1),
                                     (512, 384, 0), (512, 384, -1)]:
            with self.subTest(tile=tile, stride=stride, batch=batch):
                args = SimpleNamespace(tile_size=tile, stride=stride, batch_size=batch)
                with patch("infer_geotiff.load_model") as load_model:
                    with self.assertRaises(ValueError):
                        run_inference(args)
                    load_model.assert_not_called()

    def test_incomplete_coverage_stops_before_writing_outputs(self):
        cpu = torch.device("cpu")
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.tif"
            with rasterio.open(source, "w", driver="GTiff", width=9, height=8,
                               count=3, dtype="uint8", transform=from_origin(100, 100, 1, 1)) as dst:
                dst.write(np.zeros((3, 8, 9), dtype=np.uint8))
            args = SimpleNamespace(
                input=str(source), output_dir=directory, name="result", overwrite=False,
                tile_size=4, stride=3, batch_size=2, threshold=0.5, save_prob=True,
                rgb_bands=[1, 2, 3], mean=[0, 0, 0], std=[1, 1, 1],
                upsample_to_gsd=None, polygonize=False,
            )
            with patch("infer_geotiff.torch.cuda.is_available", return_value=True), \
                 patch("infer_geotiff.torch.device", return_value=cpu), \
                 patch("infer_geotiff.load_model"), \
                 patch("infer_geotiff.predict_batch",
                       side_effect=lambda model, batch: torch.full(batch.shape[:1] + batch.shape[-2:], 0.75)):
                with patch("infer_geotiff.compute_starts", return_value=[0]):
                    with self.assertRaisesRegex(RuntimeError, "uncovered pixels"):
                        run_inference(args)
                self.assertFalse((Path(directory) / "result_mask.tif").exists())
                self.assertFalse((Path(directory) / "result_prob.tif").exists())

                run_inference(args)
                with rasterio.open(Path(directory) / "result_mask.tif") as result:
                    np.testing.assert_array_equal(result.read(1), np.ones((8, 9), dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
