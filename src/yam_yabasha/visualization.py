"""Visual diagnostics for predictions and documentation."""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def draw_polyline(
    image_bgr: np.ndarray,
    points: np.ndarray,
    color: tuple[int, int, int] = (0, 0, 255),
    width: int = 3,
) -> np.ndarray:
    """Draw an open x/y polyline on a BGR image."""
    output = image_bgr.copy()
    points = np.asarray(points)
    if len(points) >= 2:
        cv2.polylines(output, [points.astype(np.int32)], False, color, width)
    return output


def overlay_mask(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (0, 180, 255),
    alpha: float = 0.35,
) -> np.ndarray:
    """Blend a binary mask over an image."""
    output = image_bgr.copy()
    selected = mask > 0
    color_array = np.asarray(color, dtype=np.uint8)
    output[selected] = (
        (1 - alpha) * output[selected] + alpha * color_array
    ).astype(np.uint8)
    return output


def registration_overlay(
    reference_bgr: np.ndarray,
    aligned_bgr: np.ndarray,
) -> np.ndarray:
    """Red/cyan overlay: correctly aligned structures appear neutral."""
    reference = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2GRAY)
    aligned = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2GRAY)
    output = np.zeros((*reference.shape, 3), dtype=np.uint8)
    output[..., 2] = reference
    output[..., 1] = aligned
    output[..., 0] = aligned
    return output


def save_pipeline_figure(
    path: str | Path,
    aligned_bgr: np.ndarray,
    water_probability: np.ndarray,
    land_probability: np.ndarray,
    corridor: np.ndarray,
    boundary_score: np.ndarray,
    shoreline_points: np.ndarray,
) -> Path:
    """Save a compact six-stage figure for one prediction."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    corridor_overlay = overlay_mask(aligned_bgr, corridor)
    shoreline_overlay = draw_polyline(aligned_bgr, shoreline_points)

    panels = [
        ("Aligned frame", cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB), None, None),
        ("Water probability", water_probability, "turbo", (0, 1)),
        ("Land probability", land_probability, "turbo", (0, 1)),
        ("Search corridor", cv2.cvtColor(corridor_overlay, cv2.COLOR_BGR2RGB), None, None),
        ("Water–land boundary score", boundary_score, "magma", (0, 1)),
        ("Extracted shoreline", cv2.cvtColor(shoreline_overlay, cv2.COLOR_BGR2RGB), None, None),
    ]
    figure, axes = plt.subplots(2, 3, figsize=(16, 9), dpi=120)
    for axis, (title, image, cmap, limits) in zip(axes.ravel(), panels):
        kwargs = {"cmap": cmap, "interpolation": "nearest"}
        if limits:
            kwargs.update(vmin=limits[0], vmax=limits[1])
        axis.imshow(image, **kwargs)
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return path
