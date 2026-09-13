"""Side-free 2D shoreline extraction from water/land probability maps."""

from dataclasses import dataclass

import cv2
import numpy as np

from .config import ROIContourConfig
from .roi import corridor_mask, split_polyline_by_mask


@dataclass
class ContourCandidate:
    """One connected boundary segment inside the search corridor."""

    points: np.ndarray
    score: float
    mean_evidence: float
    length_px: float
    length_fraction: float


@dataclass
class ShorelineResult:
    """Final shoreline plus intermediate maps for debugging and GUIs."""

    points: np.ndarray
    confidence: float
    candidates: list[ContourCandidate]
    corridor_mask: np.ndarray
    contrast: np.ndarray
    smooth_contrast: np.ndarray
    water_region: np.ndarray
    gradient: np.ndarray
    semantic_support: np.ndarray
    boundary_score: np.ndarray


def polyline_length(points: np.ndarray) -> float:
    """Return open-polyline arc length in pixels."""
    if len(points) < 2:
        return 0.0
    steps = np.diff(np.asarray(points, dtype=np.float64), axis=0)
    return float(np.linalg.norm(steps, axis=1).sum())


def _smooth_polyline(points: np.ndarray, window: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.int32)
    if len(points) < 3 or window <= 1:
        return points
    if window % 2 == 0:
        window += 1
    radius = window // 2
    output = points.copy()
    for index in range(len(points)):
        neighborhood = points[max(0, index - radius) : index + radius + 1]
        output[index] = np.round(np.median(neighborhood, axis=0)).astype(np.int32)
    return output


def _clip_closed_contour(
    contour: np.ndarray,
    mask: np.ndarray,
    min_points: int,
) -> list[np.ndarray]:
    """Clip a cyclic OpenCV contour without splitting a wraparound run."""
    height, width = mask.shape
    xs = np.clip(contour[:, 0], 0, width - 1)
    ys = np.clip(contour[:, 1], 0, height - 1)
    inside = mask[ys, xs] > 0
    if inside.all():
        return [contour]
    if not inside.any():
        return []
    outside_index = int(np.flatnonzero(~inside)[0])
    rotated = np.roll(contour, -outside_index, axis=0)
    return split_polyline_by_mask(rotated, mask, min_points=min_points)


def extract_shoreline(
    water_probability: np.ndarray,
    land_probability: np.ndarray,
    hint_points: np.ndarray,
    config: ROIContourConfig,
) -> ShorelineResult:
    """Extract the strongest connected water/land boundary inside a hint ROI.

    Processing order matters: contours are extracted from the full probability
    maps and clipped to the corridor only afterwards. Consequently, the
    artificial corridor edge can never become a shoreline candidate.
    """
    water = np.asarray(water_probability, dtype=np.float32)
    land = np.asarray(land_probability, dtype=np.float32)
    if water.ndim != 2 or water.shape != land.shape:
        raise ValueError("Water and land maps must be equal-size 2D arrays.")

    contrast = water - land
    smooth_contrast = (
        cv2.GaussianBlur(
            contrast,
            (0, 0),
            sigmaX=config.blur_sigma,
            sigmaY=config.blur_sigma,
        )
        if config.blur_sigma > 0
        else contrast.copy()
    )

    water_region = (smooth_contrast >= config.contrast_threshold).astype(np.uint8)
    kernel_size = max(1, int(config.morphology_kernel))
    if kernel_size % 2 == 0:
        kernel_size += 1
    morphology_kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    if config.close_iterations:
        water_region = cv2.morphologyEx(
            water_region,
            cv2.MORPH_CLOSE,
            morphology_kernel,
            iterations=config.close_iterations,
        )
    if config.open_iterations:
        water_region = cv2.morphologyEx(
            water_region,
            cv2.MORPH_OPEN,
            morphology_kernel,
            iterations=config.open_iterations,
        )

    gradient_x = cv2.Sobel(smooth_contrast, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(smooth_contrast, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gradient_x, gradient_y)
    nonzero_gradient = gradient[gradient > 0]
    gradient_scale = (
        float(np.percentile(nonzero_gradient, 99))
        if nonzero_gradient.size
        else 1.0
    )
    gradient = np.clip(gradient / max(gradient_scale, 1e-9), 0.0, 1.0)

    semantic_size = max(1, 2 * config.semantic_radius_px + 1)
    semantic_kernel = np.ones((semantic_size, semantic_size), dtype=np.uint8)
    nearby_water = cv2.dilate(water, semantic_kernel)
    nearby_land = cv2.dilate(land, semantic_kernel)
    semantic_support = np.minimum(nearby_water, nearby_land)
    boundary_score = gradient * semantic_support

    search_mask = corridor_mask(
        water.shape,
        hint_points,
        config.corridor_radius_px,
    )
    margin = config.border_margin_px
    if margin:
        search_mask[:margin] = 0
        search_mask[-margin:] = 0
        search_mask[:, :margin] = 0
        search_mask[:, -margin:] = 0

    contours, _ = cv2.findContours(
        water_region * 255,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_NONE,
    )
    hint_length = max(polyline_length(hint_points), 1.0)
    candidates: list[ContourCandidate] = []
    for raw_contour in contours:
        contour = raw_contour.reshape(-1, 2).astype(np.int32)
        runs = _clip_closed_contour(contour, search_mask, config.min_run_points)
        for run in runs:
            length = polyline_length(run)
            if length < config.min_run_length_px:
                continue
            xs = np.clip(run[:, 0], 0, water.shape[1] - 1)
            ys = np.clip(run[:, 1], 0, water.shape[0] - 1)
            evidence = float(np.mean(boundary_score[ys, xs]))
            length_fraction = min(length / hint_length, 1.0)
            score = evidence + config.length_weight * length_fraction
            candidates.append(
                ContourCandidate(
                    points=run,
                    score=score,
                    mean_evidence=evidence,
                    length_px=length,
                    length_fraction=length_fraction,
                )
            )
    candidates.sort(key=lambda candidate: candidate.score, reverse=True)

    if candidates:
        points = _smooth_polyline(candidates[0].points, config.smoothing_window)
        confidence = candidates[0].score
    else:
        points = np.empty((0, 2), dtype=np.int32)
        confidence = float("nan")

    return ShorelineResult(
        points=points,
        confidence=confidence,
        candidates=candidates,
        corridor_mask=search_mask,
        contrast=contrast,
        smooth_contrast=smooth_contrast,
        water_region=water_region * 255,
        gradient=gradient,
        semantic_support=semantic_support,
        boundary_score=boundary_score,
    )
