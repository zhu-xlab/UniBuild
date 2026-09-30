import unittest

import numpy as np

from infer_geotiff import scale_rgb


class ScaleRgbTest(unittest.TestCase):
    def test_scales_from_0_255_and_clips(self):
        rgb = np.array([[[-1, 1], [255, 1000]]] * 3, dtype=np.float32)

        scaled = scale_rgb(rgb)

        np.testing.assert_allclose(scaled[0], [[0.0, 1.0 / 255.0], [1.0, 1.0]])


if __name__ == "__main__":
    unittest.main()
