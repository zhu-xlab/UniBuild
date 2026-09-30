import unittest

import numpy as np

from infer_geotiff import scale_rgb


class ScaleRgbTest(unittest.TestCase):
    def test_auto_uses_full_band_range_for_high_range_data(self):
        rgb = np.array([[[0, 1], [2, 1000]]] * 3, dtype=np.float32)

        scaled = scale_rgb(rgb, "auto")

        np.testing.assert_allclose(scaled[0], [[0.0, 0.001], [0.002, 1.0]])


if __name__ == "__main__":
    unittest.main()
