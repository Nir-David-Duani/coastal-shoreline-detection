"""Image, polyline, and filesystem helpers."""

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_images(folder: str | Path) -> list[Path]:
    """Return supported images in deterministic filename order."""
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_image(path: str | Path, grayscale: bool = False) -> np.ndarray:
    """Load an image and raise a useful error when decoding fails."""
    path = Path(path)
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), flag)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return image


def save_image(path: str | Path, image: np.ndarray) -> Path:
    """Save an image, creating its parent directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not save image: {path}")
    return path


def load_hint(path: str | Path) -> np.ndarray:
    """Load a user-provided reference hint from JSON."""
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    points = np.asarray(payload["points"], dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        raise ValueError(f"Invalid shoreline hint: {path}")
    return points


def save_hint(path: str | Path, points: np.ndarray) -> Path:
    """Save reference-frame hint points as JSON."""
    path = Path(path)
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        raise ValueError("A shoreline hint requires at least two (x, y) points.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"points": points.tolist()}, indent=2), encoding="utf-8")
    return path


def save_polyline_csv(path: str | Path, points: np.ndarray) -> Path:
    """Save an ``(N, 2)`` shoreline as an x/y CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(np.asarray(points), columns=["x", "y"]).to_csv(path, index=False)
    return path
