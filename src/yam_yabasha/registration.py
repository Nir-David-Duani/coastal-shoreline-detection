"""SIFT/ORB feature registration to a fixed camera reference."""

from dataclasses import dataclass

import cv2
import numpy as np

from .config import RegistrationConfig


@dataclass
class RegistrationResult:
    """Aligned image, transform, diagnostics, and quality decision."""

    aligned_image: np.ndarray
    transform: np.ndarray | None
    success: bool
    metrics: dict[str, float | int | bool | str | None]
    keypoints_reference: tuple[cv2.KeyPoint, ...]
    keypoints_moving: tuple[cv2.KeyPoint, ...]
    matches: list[cv2.DMatch]
    inlier_mask: np.ndarray | None


def _detector(config: RegistrationConfig):
    method = config.feature_type.lower()
    if method == "sift":
        return cv2.SIFT_create()
    if method == "orb":
        return cv2.ORB_create(nfeatures=config.orb_features)
    if method == "akaze":
        return cv2.AKAZE_create()
    raise ValueError(f"Unsupported feature detector: {config.feature_type}")


def _detect(
    image_gray: np.ndarray,
    config: RegistrationConfig,
    mask: np.ndarray | None = None,
) -> tuple[tuple[cv2.KeyPoint, ...], np.ndarray | None]:
    keypoints, descriptors = _detector(config).detectAndCompute(image_gray, mask)
    return tuple(keypoints), descriptors


def _match(
    reference_descriptors: np.ndarray | None,
    moving_descriptors: np.ndarray | None,
    config: RegistrationConfig,
) -> list[cv2.DMatch]:
    if reference_descriptors is None or moving_descriptors is None:
        return []
    norm = (
        cv2.NORM_L2
        if config.feature_type.lower() == "sift" and reference_descriptors.dtype != np.uint8
        else cv2.NORM_HAMMING
    )
    raw = cv2.BFMatcher(norm).knnMatch(reference_descriptors, moving_descriptors, k=2)
    return [
        best
        for pair in raw
        if len(pair) == 2
        for best, second in [pair]
        if best.distance < config.ratio_test * second.distance
    ]


def _filter_reference_roi(
    keypoints: tuple[cv2.KeyPoint, ...],
    matches: list[cv2.DMatch],
    mask: np.ndarray | None,
) -> list[cv2.DMatch]:
    if mask is None:
        return matches
    height, width = mask.shape
    kept = []
    for match in matches:
        x, y = keypoints[match.queryIdx].pt
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < width and 0 <= yi < height and mask[yi, xi] > 0:
            kept.append(match)
    return kept


def _project(
    points: np.ndarray,
    transform: np.ndarray,
    transform_type: str,
) -> np.ndarray:
    if transform_type == "homography":
        return cv2.perspectiveTransform(points.reshape(-1, 1, 2), transform).reshape(-1, 2)
    homogeneous = np.column_stack([points, np.ones(len(points), dtype=np.float32)])
    return (transform @ homogeneous.T).T


def warp_image(
    image: np.ndarray,
    transform: np.ndarray,
    output_shape: tuple[int, int],
    transform_type: str,
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """Warp an image into reference coordinates."""
    height, width = output_shape
    if transform_type == "homography":
        return cv2.warpPerspective(
            image,
            transform,
            (width, height),
            flags=interpolation,
        )
    return cv2.warpAffine(
        image,
        transform,
        (width, height),
        flags=interpolation,
    )


def align_to_reference(
    reference_image: np.ndarray,
    moving_image: np.ndarray,
    config: RegistrationConfig,
    reference_roi_mask: np.ndarray | None = None,
) -> RegistrationResult:
    """Estimate moving-to-reference alignment and reject weak transforms."""
    reference_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    moving_gray = cv2.cvtColor(moving_image, cv2.COLOR_BGR2GRAY)
    kp_ref, desc_ref = _detect(reference_gray, config, reference_roi_mask)
    kp_mov, desc_mov = _detect(moving_gray, config)
    matches = _filter_reference_roi(
        kp_ref,
        _match(desc_ref, desc_mov, config),
        reference_roi_mask,
    )

    min_matches = 4 if config.transform_type == "homography" else 3
    transform = None
    inlier_mask = None
    reprojection_error = None
    if len(matches) >= min_matches:
        ref_points = np.float32([kp_ref[match.queryIdx].pt for match in matches])
        mov_points = np.float32([kp_mov[match.trainIdx].pt for match in matches])
        if config.transform_type == "homography":
            transform, inlier_mask = cv2.findHomography(
                mov_points,
                ref_points,
                cv2.RANSAC,
                config.ransac_threshold_px,
            )
        else:
            transform, inlier_mask = cv2.estimateAffinePartial2D(
                mov_points,
                ref_points,
                method=cv2.RANSAC,
                ransacReprojThreshold=config.ransac_threshold_px,
            )
        if transform is not None and inlier_mask is not None:
            inliers = inlier_mask.ravel().astype(bool)
            projected = _project(mov_points[inliers], transform, config.transform_type)
            reprojection_error = float(
                np.mean(np.linalg.norm(projected - ref_points[inliers], axis=1))
            )

    num_inliers = int(np.sum(inlier_mask)) if inlier_mask is not None else 0
    inlier_ratio = num_inliers / len(matches) if matches else 0.0
    quality_ok = (
        transform is not None
        and num_inliers >= config.min_inliers
        and inlier_ratio >= config.min_inlier_ratio
        and reprojection_error is not None
        and reprojection_error <= config.max_reprojection_error_px
    )
    if not quality_ok:
        transform = None

    aligned = (
        warp_image(
            moving_image,
            transform,
            reference_image.shape[:2],
            config.transform_type,
        )
        if transform is not None
        else moving_image.copy()
    )
    metrics = {
        "registration_success": quality_ok,
        "feature_type": config.feature_type,
        "transform_type": config.transform_type,
        "reference_roi_used": reference_roi_mask is not None,
        "num_keypoints_reference": len(kp_ref),
        "num_keypoints_moving": len(kp_mov),
        "num_matches": len(matches),
        "num_inliers": num_inliers,
        "inlier_ratio": float(inlier_ratio),
        "reprojection_error_px": reprojection_error,
    }
    return RegistrationResult(
        aligned_image=aligned,
        transform=transform,
        success=quality_ok,
        metrics=metrics,
        keypoints_reference=kp_ref,
        keypoints_moving=kp_mov,
        matches=matches,
        inlier_mask=inlier_mask,
    )
