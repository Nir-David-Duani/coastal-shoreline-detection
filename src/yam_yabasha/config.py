"""Typed configuration for the shoreline pipeline."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


CombineMode = Literal["max", "mean"]
TransformType = Literal["homography", "affine"]


@dataclass(frozen=True)
class RegistrationConfig:
    """Feature-based alignment to a fixed reference frame."""

    feature_type: str = "sift"
    transform_type: TransformType = "homography"
    ratio_test: float = 0.75
    ransac_threshold_px: float = 5.0
    orb_features: int = 5000
    min_inliers: int = 8
    min_inlier_ratio: float = 0.25
    max_reprojection_error_px: float = 5.0


@dataclass(frozen=True)
class SegmentationConfig:
    """CLIPSeg model and prompt ensemble."""

    model_name: str = "CIDAS/clipseg-rd64-refined"
    water_prompts: tuple[str, ...] = (
        "water",
        "sea water",
        "foamy water",
        "water with foam",
        "waves on water",
    )
    land_prompts: tuple[str, ...] = (
        "sand",
        "beach",
        "wet sand",
        "rocky shore",
        "land",
        "coast",
    )
    water_combine: CombineMode = "max"
    land_combine: CombineMode = "max"
    device: str | None = None


@dataclass(frozen=True)
class ROIContourConfig:
    """Side-free 2D water/land contour extraction."""

    corridor_radius_px: float = 120.0
    contrast_threshold: float = -0.05
    blur_sigma: float = 2.0
    morphology_kernel: int = 5
    close_iterations: int = 2
    open_iterations: int = 1
    semantic_radius_px: int = 7
    min_run_points: int = 20
    min_run_length_px: float = 50.0
    length_weight: float = 0.25
    smoothing_window: int = 7
    border_margin_px: int = 2


@dataclass(frozen=True)
class PipelineConfig:
    """Filesystem and algorithm settings."""

    project_root: Path = Path(".")
    data_dir: Path = Path("data")
    output_dir: Path = Path("outputs")
    registration: RegistrationConfig = field(default_factory=RegistrationConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    contour: ROIContourConfig = field(default_factory=ROIContourConfig)

    @property
    def resolved_data_dir(self) -> Path:
        return self.data_dir if self.data_dir.is_absolute() else self.project_root / self.data_dir

    @property
    def resolved_output_dir(self) -> Path:
        return self.output_dir if self.output_dir.is_absolute() else self.project_root / self.output_dir
