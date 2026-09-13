"""Regenerate evidence-based README figures from labeled project images."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from yam_yabasha import PipelineConfig, ShorelinePipeline
from yam_yabasha.io import list_images, load_image
from yam_yabasha.pipeline import PredictionResult
from yam_yabasha.registration import RegistrationResult, align_to_reference, warp_image
from yam_yabasha.visualization import (
    draw_polyline,
    overlay_mask,
    registration_overlay,
)


@dataclass
class LabeledExample:
    """A prediction paired with its registered manual water mask."""

    result: PredictionResult
    ground_truth_boundary: np.ndarray
    median_error_px: float


def _rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _mask_boundary(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 127).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    return cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel) > 0


def _prediction_error(points: np.ndarray, boundary: np.ndarray) -> float:
    if len(points) == 0 or not np.any(boundary):
        return float("inf")
    distance = cv2.distanceTransform(
        (~boundary).astype(np.uint8),
        cv2.DIST_L2,
        5,
    )
    xs = np.clip(points[:, 0], 0, distance.shape[1] - 1)
    ys = np.clip(points[:, 1], 0, distance.shape[0] - 1)
    return float(np.median(distance[ys, xs]))


def _labeled_pairs(root: Path, location: str) -> list[tuple[Path, Path]]:
    pairs = []
    for image_path in list_images(root / "data" / location / "images"):
        task = int(image_path.stem.replace("task", ""))
        mask_path = root / "data" / location / "masks" / f"task-{task:04d}.png"
        if mask_path.exists():
            pairs.append((image_path, mask_path))
    return pairs


def _best_labeled_example(
    root: Path,
    pipeline: ShorelinePipeline,
    location: str,
) -> LabeledExample:
    candidates = []
    for image_path, mask_path in _labeled_pairs(root, location):
        result = pipeline.process(
            image_path,
            location,
            register=True,
            save_outputs=False,
        )
        mask = load_image(mask_path, grayscale=True)
        if result.registration and result.registration.transform is not None:
            mask = warp_image(
                mask,
                result.registration.transform,
                result.aligned_image.shape[:2],
                pipeline.config.registration.transform_type,
                interpolation=cv2.INTER_NEAREST,
            )
        boundary = _mask_boundary(mask)
        boundary &= result.shoreline.corridor_mask > 0
        error = _prediction_error(result.shoreline.points, boundary)
        print(
            f"{location}/{image_path.name}: "
            f"median labeled-boundary error={error:.2f}px"
        )
        candidates.append(LabeledExample(result, boundary, error))
    valid = [
        candidate
        for candidate in candidates
        if np.isfinite(candidate.median_error_px)
        and len(candidate.result.shoreline.points) >= 100
    ]
    if not valid:
        raise RuntimeError(f"No valid labeled README example for {location}")
    return min(valid, key=lambda candidate: candidate.median_error_px)


def _ground_truth_overlay(example: LabeledExample) -> np.ndarray:
    output = example.result.aligned_image.copy()
    thick_boundary = cv2.dilate(
        example.ground_truth_boundary.astype(np.uint8),
        np.ones((3, 3), dtype=np.uint8),
    )
    output[thick_boundary > 0] = (0, 255, 0)
    return draw_polyline(output, example.result.shoreline.points, (0, 0, 255), 4)


def _save_pipeline_figure(path: Path, example: LabeledExample) -> None:
    result = example.result
    segmentation = result.segmentation
    shoreline = result.shoreline
    corridor_view = overlay_mask(result.aligned_image, shoreline.corridor_mask)
    final_overlay = _ground_truth_overlay(example)
    stages = [
        ("Aligned frame", _rgb(result.aligned_image), None, None),
        ("Water probability", segmentation.water_probability, "turbo", (0, 1)),
        ("Land probability", segmentation.land_probability, "turbo", (0, 1)),
        ("Water - land contrast", segmentation.contrast, "coolwarm", (-1, 1)),
        ("Thresholded water region", shoreline.water_region, "gray", (0, 255)),
        ("Search corridor", _rgb(corridor_view), None, None),
        ("Semantic boundary score", shoreline.boundary_score, "magma", (0, 1)),
        (
            f"Prediction (red) vs label (green)\n"
            f"median error={example.median_error_px:.1f}px",
            _rgb(final_overlay),
            None,
            None,
        ),
    ]
    figure, axes = plt.subplots(2, 4, figsize=(19, 9), dpi=120)
    for axis, (title, image, cmap, limits) in zip(axes.ravel(), stages):
        kwargs = {"cmap": cmap, "interpolation": "nearest"}
        if limits:
            kwargs.update(vmin=limits[0], vmax=limits[1])
        axis.imshow(image, **kwargs)
        axis.set_title(title)
        axis.axis("off")
    figure.suptitle(
        f"{result.location.title()} pipeline — {result.image_path.name}",
        fontsize=16,
    )
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def _corner_displacement(
    transform: np.ndarray,
    moving_shape: tuple[int, int],
    reference_shape: tuple[int, int],
) -> float:
    moving_height, moving_width = moving_shape
    reference_height, reference_width = reference_shape
    moving_corners = np.float32(
        [
            [0, 0],
            [moving_width - 1, 0],
            [moving_width - 1, moving_height - 1],
            [0, moving_height - 1],
        ]
    )
    expected = np.float32(
        [
            [0, 0],
            [reference_width - 1, 0],
            [reference_width - 1, reference_height - 1],
            [0, reference_height - 1],
        ]
    )
    if transform.shape == (3, 3):
        projected = cv2.perspectiveTransform(
            moving_corners.reshape(-1, 1, 2),
            transform,
        ).reshape(-1, 2)
    else:
        projected = cv2.transform(
            moving_corners.reshape(-1, 1, 2),
            transform,
        ).reshape(-1, 2)
    return float(np.mean(np.linalg.norm(projected - expected, axis=1)))


def _strong_registration_example(
    root: Path,
    config: PipelineConfig,
) -> tuple[np.ndarray, np.ndarray, RegistrationResult, float, Path]:
    location = "entrance"
    reference = load_image(root / "data" / location / "reference.jpg")
    images = list_images(root / "Test" / location)
    if not images:
        raise FileNotFoundError("README registration asset requires Test/entrance")
    best = None
    for image_path in images:
        moving = load_image(image_path)
        registration = align_to_reference(reference, moving, config.registration)
        if not registration.success or registration.transform is None:
            continue
        displacement = _corner_displacement(
            registration.transform,
            moving.shape[:2],
            reference.shape[:2],
        )
        if best is None or displacement > best[3]:
            best = (reference, moving, registration, displacement, image_path)
    if best is None:
        raise RuntimeError("No successful registration example")
    return best


def _save_registration_figure(
    path: Path,
    reference: np.ndarray,
    moving: np.ndarray,
    registration: RegistrationResult,
    displacement: float,
    image_path: Path,
) -> None:
    inlier_mask = (
        registration.inlier_mask.ravel().astype(bool)
        if registration.inlier_mask is not None
        else np.zeros(len(registration.matches), dtype=bool)
    )
    inlier_matches = [
        match
        for match, is_inlier in zip(registration.matches, inlier_mask)
        if is_inlier
    ]
    if len(inlier_matches) > 80:
        indices = np.linspace(0, len(inlier_matches) - 1, 80, dtype=int)
        inlier_matches = [inlier_matches[index] for index in indices]
    match_view = cv2.drawMatches(
        reference,
        registration.keypoints_reference,
        moving,
        registration.keypoints_moving,
        inlier_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    naive_moving = cv2.resize(
        moving,
        (reference.shape[1], reference.shape[0]),
        interpolation=cv2.INTER_AREA,
    )
    before_overlay = registration_overlay(reference, naive_moving)
    after_overlay = registration_overlay(reference, registration.aligned_image)
    valid_region = warp_image(
        np.full(moving.shape[:2], 255, dtype=np.uint8),
        registration.transform,
        reference.shape[:2],
        "homography" if registration.transform.shape == (3, 3) else "affine",
        interpolation=cv2.INTER_NEAREST,
    )
    after_overlay[valid_region == 0] = 0
    stages = [
        ("Fixed reference", _rgb(reference)),
        (f"Incoming frame\n{image_path.name}", _rgb(moving)),
        (f"SIFT + RANSAC inliers ({len(inlier_matches)} shown)", _rgb(match_view)),
        ("Before alignment\nred/cyan disagreement", _rgb(before_overlay)),
        ("Frame after homography", _rgb(registration.aligned_image)),
        ("After alignment\nstructures overlap", _rgb(after_overlay)),
    ]
    figure, axes = plt.subplots(2, 3, figsize=(19, 10), dpi=120)
    for axis, (title, image) in zip(axes.ravel(), stages):
        axis.imshow(image)
        axis.set_title(title)
        axis.axis("off")
    metrics = registration.metrics
    figure.suptitle(
        "Feature registration to the reference coordinate system\n"
        f"corner displacement={displacement:.1f}px, "
        f"inliers={metrics['num_inliers']}, "
        f"reprojection error={metrics['reprojection_error_px']:.2f}px",
        fontsize=15,
    )
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def _save_location_hero(path: Path, examples: list[LabeledExample]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=140)
    for axis, example in zip(axes, examples):
        result = example.result
        overlay = draw_polyline(
            result.aligned_image,
            result.shoreline.points,
            color=(0, 0, 255),
            width=4,
        )
        axis.imshow(_rgb(overlay))
        axis.set_title(
            f"{result.location.title()} — {result.image_path.stem}\n"
            f"selected shoreline"
        )
        axis.axis("off")
    figure.suptitle("Shoreline extraction at two fixed-camera locations", fontsize=16)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    docs_images = root / "docs" / "images"
    docs_images.mkdir(parents=True, exist_ok=True)
    config = PipelineConfig(project_root=root)
    pipeline = ShorelinePipeline(config)

    examples = [
        _best_labeled_example(root, pipeline, location)
        for location in ("entrance", "porch")
    ]
    for example in examples:
        _save_pipeline_figure(
            docs_images / f"{example.result.location}_pipeline.png",
            example,
        )
    _save_location_hero(docs_images / "location_examples.png", examples)

    reference, moving, registration, displacement, image_path = (
        _strong_registration_example(root, config)
    )
    _save_registration_figure(
        docs_images / "registration_sift_alignment.png",
        reference,
        moving,
        registration,
        displacement,
        image_path,
    )


if __name__ == "__main__":
    main()
