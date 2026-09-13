import cv2
import numpy as np

from yam_yabasha.registration import warp_image


def test_homography_warp_uses_reference_output_shape():
    image = np.zeros((30, 40), dtype=np.uint8)
    image[10, 10] = 255
    transform = np.array(
        [[1, 0, 5], [0, 1, 3], [0, 0, 1]],
        dtype=np.float64,
    )

    warped = warp_image(
        image,
        transform,
        output_shape=(50, 60),
        transform_type="homography",
        interpolation=cv2.INTER_NEAREST,
    )

    assert warped.shape == (50, 60)
    assert warped[13, 15] == 255
