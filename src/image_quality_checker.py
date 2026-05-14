"""
Basic medical image quality checks.

These checks identify quality concerns only. They do not diagnose image content.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageStat


def check_image_quality(path: str | Path) -> tuple[list[str], dict]:
    image_path = Path(path)
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        width, height = image.size
        grayscale = image.convert("L")
        pixels = np.asarray(grayscale, dtype=np.float32)

        brightness = float(pixels.mean())
        contrast = float(pixels.std())

        # Laplacian variance is a common lightweight blur heuristic.
        try:
            import cv2

            blur_score = float(cv2.Laplacian(pixels, cv2.CV_64F).var())
        except Exception:
            blur_score = contrast

        warnings: list[str] = []
        if width < 500 or height < 500:
            warnings.append("Low resolution image warning; clinician review may need a clearer image.")
        if blur_score < 70:
            warnings.append("Blurry image warning; fine details may not be reliable.")
        if brightness < 55:
            warnings.append("Dark image warning; image quality concern detected.")
        if brightness > 215:
            warnings.append("Very bright image warning; image quality concern detected.")
        if contrast < 18:
            warnings.append("Low contrast image warning; image may be difficult to interpret.")

        metadata = {
            "width": width,
            "height": height,
            "mode": image.mode,
            "brightness": round(brightness, 2),
            "contrast": round(contrast, 2),
            "blur_score": round(blur_score, 2),
        }
        stat = ImageStat.Stat(image)
        metadata["channel_mean"] = [round(value, 2) for value in stat.mean]

    return warnings, metadata

