import numpy as np

from yam_yabasha.io import load_hint, save_hint
from yam_yabasha.roi import corridor_mask, split_polyline_by_mask


def test_corridor_contains_hint_and_excludes_far_pixels():
    hint = np.array([[20, 50], [80, 50]])
    mask = corridor_mask((100, 100), hint, radius_px=10)

    assert mask[50, 50] == 255
    assert mask[10, 50] == 0


def test_split_polyline_keeps_connected_inside_runs():
    points = np.column_stack([np.arange(10), np.full(10, 5)])
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[:, 2:5] = 255
    mask[:, 7:9] = 255

    runs = split_polyline_by_mask(points, mask)

    assert [len(run) for run in runs] == [3, 2]


def test_hint_json_roundtrip(tmp_path):
    expected = np.array([[1.25, 2.5], [9.0, 10.75]])
    path = tmp_path / "hint.json"

    save_hint(path, expected)

    np.testing.assert_allclose(load_hint(path), expected)
