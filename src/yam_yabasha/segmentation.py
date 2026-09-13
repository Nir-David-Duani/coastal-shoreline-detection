"""CLIPSeg water/land prompt inference."""

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as functional
from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

from .config import SegmentationConfig


@dataclass
class SegmentationResult:
    """Per-prompt and combined full-resolution probability maps."""

    water_probability: np.ndarray
    land_probability: np.ndarray
    contrast: np.ndarray
    water_prompt_maps: dict[str, np.ndarray]
    land_prompt_maps: dict[str, np.ndarray]


def combine_maps(maps: np.ndarray, mode: str) -> np.ndarray:
    """Combine prompt maps with a conservative mean or inclusive maximum."""
    if maps.ndim != 3 or len(maps) == 0:
        raise ValueError("Prompt maps must have shape (prompts, height, width).")
    if mode == "max":
        return maps.max(axis=0)
    if mode == "mean":
        return maps.mean(axis=0)
    raise ValueError(f"Unsupported prompt combination: {mode}")


class CLIPSegSegmenter:
    """Load CLIPSeg once and reuse it across images."""

    def __init__(self, config: SegmentationConfig):
        self.config = config
        self.device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = CLIPSegProcessor.from_pretrained(config.model_name)
        self.model = CLIPSegForImageSegmentation.from_pretrained(config.model_name)
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def predict_prompt_maps(
        self,
        image_bgr: np.ndarray,
        prompts: tuple[str, ...] | list[str],
    ) -> np.ndarray:
        """Return one probability map per prompt at input-image resolution."""
        if not prompts:
            raise ValueError("At least one prompt is required.")
        image = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        inputs = self.processor(
            text=list(prompts),
            images=[image] * len(prompts),
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        logits = self.model(**inputs).logits
        if logits.ndim == 2:
            logits = logits.unsqueeze(0)
        probabilities = torch.sigmoid(logits).unsqueeze(1)
        probabilities = functional.interpolate(
            probabilities,
            size=image_bgr.shape[:2],
            mode="bilinear",
            align_corners=False,
        )[:, 0]
        return probabilities.cpu().numpy().astype(np.float32)

    def predict(self, image_bgr: np.ndarray) -> SegmentationResult:
        """Predict water, land, and their signed contrast in one model call."""
        water_prompts = self.config.water_prompts
        land_prompts = self.config.land_prompts
        prompts = water_prompts + land_prompts
        maps = self.predict_prompt_maps(image_bgr, prompts)
        water_stack = maps[: len(water_prompts)]
        land_stack = maps[len(water_prompts) :]
        water = combine_maps(water_stack, self.config.water_combine)
        land = combine_maps(land_stack, self.config.land_combine)
        return SegmentationResult(
            water_probability=water,
            land_probability=land,
            contrast=water - land,
            water_prompt_maps=dict(zip(water_prompts, water_stack)),
            land_prompt_maps=dict(zip(land_prompts, land_stack)),
        )
