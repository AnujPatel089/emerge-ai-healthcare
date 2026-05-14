"""
Medical image upload assessment helpers.

The assessment reports metadata, quality concerns, and optional OCR text. It
does not provide a diagnosis.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PIL import Image

from src.image_quality_checker import check_image_quality


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def extract_image_ocr(path: str | Path) -> tuple[bool, str]:
    try:
        import pytesseract
    except Exception:
        return False, "OCR is not available. Image preview and metadata analysis completed."

    try:
        with Image.open(path) as image:
            text = pytesseract.image_to_string(image)
        return True, text.strip()
    except Exception as exc:
        return False, f"OCR could not process this image: {exc}"


def assess_medical_image(path: str | Path, upload_date: str | None = None) -> dict:
    image_path = Path(path)
    with Image.open(image_path) as image:
        width, height = image.size
        image_format = image.format or image_path.suffix.replace(".", "").upper()

    quality_notes, quality_metrics = check_image_quality(image_path)
    ocr_available, ocr_text = extract_image_ocr(image_path)

    metadata = {
        "file_name": image_path.name,
        "file_size_bytes": image_path.stat().st_size,
        "image_width": width,
        "image_height": height,
        "image_format": image_format,
        "upload_date": upload_date or datetime.utcnow().isoformat(timespec="seconds"),
        **quality_metrics,
    }

    return {
        "image_metadata": metadata,
        "image_quality_notes": quality_notes,
        "ocr_available": ocr_available,
        "ocr_text": ocr_text if ocr_available else "",
        "ocr_message": "" if ocr_available else ocr_text,
    }

