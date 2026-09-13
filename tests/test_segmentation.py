import numpy as np
import pytest

from yam_yabasha.segmentation import combine_maps


def test_prompt_map_combinations():
    maps = np.array([np.zeros((3, 4)), np.ones((3, 4))], dtype=np.float32)

    np.testing.assert_allclose(combine_maps(maps, "max"), 1.0)
    np.testing.assert_allclose(combine_maps(maps, "mean"), 0.5)


def test_unknown_prompt_combination_is_rejected():
    with pytest.raises(ValueError, match="Unsupported"):
        combine_maps(np.ones((2, 3, 4), dtype=np.float32), "median")
