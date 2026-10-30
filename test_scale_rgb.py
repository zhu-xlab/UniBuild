import unittest

import numpy as np

from infer_geotiff import scale_rgb


class ScaleRgbTest(unittest.TestCase):
    def test_scales_from_0_255(self):
        rgb = np.array([[[0, 1], [255, 128]]] * 3, dtype=np.float32)

        scaled = scale_rgb(rgb)

        np.testing.assert_allclose(scaled[0], [[0.0, 1.0 / 255.0], [1.0, 128.0 / 255.0]])

    def test_rejects_nonfinite_and_out_of_range_values(self):
        for value in [np.nan, np.inf, -np.inf, -1, 256]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                scale_rgb(np.array([value], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
