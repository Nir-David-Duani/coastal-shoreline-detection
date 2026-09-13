import numpy as np

from yam_yabasha.config import ROIContourConfig
from yam_yabasha.shoreline import _clip_closed_contour, extract_shoreline


def test_validated_default_contrast_threshold():
    assert ROIContourConfig().contrast_threshold == -0.05


def _semantic_maps(boundary_y: np.ndarray, height: int) -> tuple[np.ndarray, np.ndarray]:
    rows = np.arange(height, dtype=np.float32)[:, None]
    signed_distance = rows - boundary_y[None, :]
    water = 1.0 / (1.0 + np.exp(-signed_distance / 2.0))
    return water.astype(np.float32), (1.0 - water).astype(np.float32)


def _test_config() -> ROIContourConfig:
    return ROIContourConfig(
        corridor_radius_px=30,
        blur_sigma=1,
        morphology_kernel=3,
        close_iterations=1,
        open_iterations=0,
        semantic_radius_px=4,
        min_run_points=15,
        min_run_length_px=25,
        smoothing_window=5,
        border_margin_px=1,
    )


def test_closed_contour_preserves_run_across_array_boundary():
    contour = np.array(
        [[2, 5], [3, 5], [7, 5], [8, 5], [4, 5], [1, 5]],
        dtype=np.int32,
    )
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[:, :5] = 255

    runs = _clip_closed_contour(contour, mask, min_points=2)

    assert len(runs) == 1
    assert len(runs[0]) == 4


def test_extracts_curved_boundary_without_direction_assumption():
    height, width = 180, 260
    x = np.arange(width)
    expected_y = 85 + 22 * np.sin(x / 42)
    water, land = _semantic_maps(expected_y, height)
    hint_x = np.arange(10, width - 10, 20)
    hint = np.column_stack([hint_x, 90 + 15 * np.sin(hint_x / 42)])

    result = extract_shoreline(water, land, hint, _test_config())

    assert len(result.points) > 100
    predicted_error = np.abs(
        result.points[:, 1] - expected_y[result.points[:, 0]]
    )
    assert np.median(predicted_error) < 3
    assert len(result.candidates) >= 1


def test_roi_border_is_not_returned_as_a_candidate():
    height, width = 140, 220
    expected_y = np.full(width, 70.0)
    water, land = _semantic_maps(expected_y, height)
    hint = np.array([[20, 70], [200, 70]])

    result = extract_shoreline(water, land, hint, _test_config())

    assert len(result.points) > 100
    assert abs(float(np.median(result.points[:, 1])) - 70) < 3


def test_returns_empty_result_when_corridor_has_no_boundary():
    height, width = 140, 220
    expected_y = np.full(width, 100.0)
    water, land = _semantic_maps(expected_y, height)
    hint = np.array([[20, 20], [200, 20]])
    config = ROIContourConfig(
        corridor_radius_px=10,
        min_run_points=10,
        min_run_length_px=20,
    )

    result = extract_shoreline(water, land, hint, config)

    assert result.points.shape == (0, 2)
    assert np.isnan(result.confidence)
