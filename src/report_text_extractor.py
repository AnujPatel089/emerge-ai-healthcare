"""
Safe text extraction for uploaded historical medical reports.

Extraction failures are returned as user-friendly messages so uploads do not
crash the app.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".csv", ".docx"}


def extract_text_from_file(path: str | Path) -> tuple[bool, str]:
    file_path = Path(path)
    suffix = file_path.suffix.lower()

    if suffix not in SUPPORTED_EXTENSIONS:
        return False, f"Unsupported file type: {suffix}"

    try:
        if suffix == ".txt":
            return True, file_path.read_text(encoding="utf-8", errors="ignore")

        if suffix == ".csv":
            df = pd.read_csv(file_path)
            return True, df.astype(str).to_csv(index=False)

        if suffix == ".pdf":
            try:
                import PyPDF2
            except Exception:
                return False, "PDF extraction requires PyPDF2. Install dependencies from requirements.txt."

            text_parts: list[str] = []
            with file_path.open("rb") as handle:
                reader = PyPDF2.PdfReader(handle)
                for page in reader.pages:
                    text_parts.append(page.extract_text() or "")
            return True, "\n".join(text_parts).strip()

        if suffix == ".docx":
            try:
                from docx import Document
            except Exception:
                return False, "DOCX extraction requires python-docx. Install dependencies from requirements.txt."

            document = Document(str(file_path))
            return True, "\n".join(paragraph.text for paragraph in document.paragraphs).strip()

    except Exception as exc:
        return False, f"Could not extract text from this report: {exc}"

    return False, "Could not extract text from this report."

