"""Yam Yabasha shoreline extraction package."""

from .config import PipelineConfig, ROIContourConfig, RegistrationConfig, SegmentationConfig
from .pipeline import ShorelinePipeline

__all__ = [
    "PipelineConfig",
    "ROIContourConfig",
    "RegistrationConfig",
    "SegmentationConfig",
    "ShorelinePipeline",
]

__version__ = "0.1.0"
