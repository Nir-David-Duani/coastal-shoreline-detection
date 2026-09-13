"""Reference-frame shoreline search regions."""

import cv2
import numpy as np


def corridor_mask(
    shape: tuple[int, int],
    hint_points: np.ndarray,
    radius_px: float,
) -> np.ndarray:
    """Create a broad binary corridor around a user hint.

    The hint only restricts *where* to search. Its tangent and shape are not
    used by the contour extractor.
    """
    points = np.asarray(hint_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        raise ValueError("hint_points must contain at least two (x, y) points")
    if radius_px <= 0:
        raise ValueError("radius_px must be positive")
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.polylines(
        mask,
        [np.round(points).astype(np.int32)],
        isClosed=False,
        color=255,
        thickness=max(1, int(round(2 * radius_px))),
    )
    return mask


def split_polyline_by_mask(
    points: np.ndarray,
    mask: np.ndarray,
    min_points: int = 2,
) -> list[np.ndarray]:
    """Split an ordered polyline into contiguous runs inside ``mask``."""
    points = np.asarray(points, dtype=np.int32)
    height, width = mask.shape[:2]
    runs: list[np.ndarray] = []
    current: list[np.ndarray] = []
    for point in points:
        x, y = map(int, point)
        inside = 0 <= x < width and 0 <= y < height and mask[y, x] > 0
        if inside:
            current.append(point)
        else:
            if len(current) >= min_points:
                runs.append(np.asarray(current, dtype=np.int32))
            current = []
    if len(current) >= min_points:
        runs.append(np.asarray(current, dtype=np.int32))
    return runs
