"""End-to-end registration, segmentation, and shoreline extraction."""

from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from .config import PipelineConfig
from .io import list_images, load_hint, load_image, save_image, save_polyline_csv
from .registration import RegistrationResult, align_to_reference
from .segmentation import CLIPSegSegmenter, SegmentationResult
from .shoreline import ShorelineResult, extract_shoreline
from .visualization import save_pipeline_figure


@dataclass
class PredictionResult:
    """All outputs for one frame."""

    location: str
    image_path: Path
    reference_path: Path
    aligned_image: np.ndarray
    registration: RegistrationResult | None
    segmentation: SegmentationResult
    shoreline: ShorelineResult
    output_paths: dict[str, Path]


class ShorelinePipeline:
    """Reusable shoreline pipeline with a lazily loaded CLIPSeg model."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._segmenter: CLIPSegSegmenter | None = None

    @property
    def segmenter(self) -> CLIPSegSegmenter:
        if self._segmenter is None:
            self._segmenter = CLIPSegSegmenter(self.config.segmentation)
        return self._segmenter

    def _location_dir(self, location: str) -> Path:
        return self.config.resolved_data_dir / location

    def _assets(self, location: str) -> tuple[Path, Path, Path]:
        location_dir = self._location_dir(location)
        reference = location_dir / "reference.jpg"
        hint = location_dir / "shoreline_hint.json"
        registration_roi = location_dir / "reference_roi_mask.png"
        if not reference.exists():
            raise FileNotFoundError(f"Missing reference image: {reference}")
        if not hint.exists():
            raise FileNotFoundError(f"Missing shoreline hint: {hint}")
        return reference, hint, registration_roi

    def _output_paths(self, location: str, image_path: Path) -> dict[str, Path]:
        root = self.config.resolved_output_dir
        stem = image_path.stem
        return {
            "aligned": root / location / stem / "aligned.jpg",
            "water_probability": root / location / stem / "water_probability.png",
            "land_probability": root / location / stem / "land_probability.png",
            "contrast": root / location / stem / "contrast.png",
            "boundary_score": root / location / stem / "boundary_score.png",
            "corridor": root / location / stem / "corridor.png",
            "shoreline_csv": root / location / stem / "shoreline.csv",
            "figure": root / location / stem / "pipeline.png",
            "metadata": root / location / stem / "metadata.json",
        }

    def process(
        self,
        image_path: str | Path,
        location: str,
        *,
        register: bool = True,
        save_outputs: bool = True,
    ) -> PredictionResult:
        """Run the complete pipeline for one image."""
        image_path = Path(image_path)
        reference_path, hint_path, registration_roi_path = self._assets(location)
        image = load_image(image_path)
        reference = load_image(reference_path)
        hint_points = load_hint(hint_path)

        registration = None
        if register:
            registration_roi = (
                load_image(registration_roi_path, grayscale=True)
                if registration_roi_path.exists()
                else None
            )
            registration = align_to_reference(
                reference,
                image,
                self.config.registration,
                registration_roi,
            )
            if not registration.success:
                raise RuntimeError(
                    f"Registration failed for {image_path}. "
                    "The reference-frame shoreline corridor was not applied."
                )
            aligned = registration.aligned_image
        else:
            if image.shape[:2] != reference.shape[:2]:
                raise ValueError(
                    "Registration can only be skipped when image and reference "
                    "have equal dimensions."
                )
            aligned = image.copy()

        segmentation = self.segmenter.predict(aligned)
        shoreline = extract_shoreline(
            segmentation.water_probability,
            segmentation.land_probability,
            hint_points,
            self.config.contour,
        )
        output_paths = self._output_paths(location, image_path)
        result = PredictionResult(
            location=location,
            image_path=image_path,
            reference_path=reference_path,
            aligned_image=aligned,
            registration=registration,
            segmentation=segmentation,
            shoreline=shoreline,
            output_paths=output_paths,
        )
        if save_outputs:
            self._save(result)
        return result

    def process_folder(
        self,
        folder: str | Path,
        location: str,
        *,
        register: bool = True,
        max_images: int | None = None,
    ) -> list[PredictionResult]:
        """Process a folder while reusing the loaded model."""
        images = list_images(folder)
        if max_images is not None:
            images = images[:max_images]
        return [
            self.process(
                path,
                location,
                register=register,
            )
            for path in tqdm(images, desc=f"Shoreline {location}")
        ]

    def _save(self, result: PredictionResult) -> None:
        paths = result.output_paths
        segmentation = result.segmentation
        shoreline = result.shoreline
        save_image(paths["aligned"], result.aligned_image)
        save_image(
            paths["water_probability"],
            np.clip(segmentation.water_probability * 255, 0, 255).astype(np.uint8),
        )
        save_image(
            paths["land_probability"],
            np.clip(segmentation.land_probability * 255, 0, 255).astype(np.uint8),
        )
        save_image(
            paths["contrast"],
            np.clip((segmentation.contrast + 1) * 127.5, 0, 255).astype(np.uint8),
        )
        save_image(
            paths["boundary_score"],
            np.clip(shoreline.boundary_score * 255, 0, 255).astype(np.uint8),
        )
        save_image(paths["corridor"], shoreline.corridor_mask)
        save_polyline_csv(paths["shoreline_csv"], shoreline.points)
        save_pipeline_figure(
            paths["figure"],
            result.aligned_image,
            segmentation.water_probability,
            segmentation.land_probability,
            shoreline.corridor_mask,
            shoreline.boundary_score,
            shoreline.points,
        )

        registration_metrics = (
            result.registration.metrics
            if result.registration is not None
            else {"registration_success": None, "registration_skipped": True}
        )
        confidence = result.shoreline.confidence
        metadata = {
            "location": result.location,
            "image": str(result.image_path),
            "reference": str(result.reference_path),
            "registration": registration_metrics,
            "shoreline": {
                "confidence": confidence if np.isfinite(confidence) else None,
                "num_points": len(result.shoreline.points),
                "num_candidates": len(result.shoreline.candidates),
            },
        }
        paths["metadata"].parent.mkdir(parents=True, exist_ok=True)
        paths["metadata"].write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )
