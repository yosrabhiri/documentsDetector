"""First processing stage: inspect a PDF before running OCR."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fitz


DEFAULT_MIN_TEXT_CHARACTERS = 20


class PdfAnalysisError(ValueError):
    """Raised when a file cannot be analyzed as a PDF."""


def analyze_pdf(
    pdf_path: str | Path,
    *,
    min_text_characters: int = DEFAULT_MIN_TEXT_CHARACTERS,
) -> dict[str, Any]:
    """Return document and page information needed before the OCR stage."""
    path = Path(pdf_path).expanduser().resolve()

    if min_text_characters < 0:
        raise ValueError("min_text_characters must be zero or greater")
    if not path.is_file():
        raise PdfAnalysisError(f"PDF file not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise PdfAnalysisError(f"Expected a .pdf file: {path}")

    try:
        document = fitz.open(path)
    except (fitz.FileDataError, RuntimeError) as exc:
        raise PdfAnalysisError(f"Invalid or unreadable PDF: {path}") from exc

    try:
        if document.needs_pass:
            raise PdfAnalysisError("Password-protected PDFs are not supported yet")

        pages = []
        for index, page in enumerate(document):
            extracted_text = page.get_text("text").strip()
            useful_characters = sum(not char.isspace() for char in extracted_text)
            pages.append(
                {
                    "page_number": index + 1,
                    "width_points": round(page.rect.width, 2),
                    "height_points": round(page.rect.height, 2),
                    "text_characters": useful_characters,
                    "image_count": len(page.get_images(full=True)),
                    "ocr_required": useful_characters < min_text_characters,
                }
            )

        ocr_pages = [page["page_number"] for page in pages if page["ocr_required"]]
        return {
            "source_file": str(path),
            "reference": path.stem,
            "file_size_bytes": path.stat().st_size,
            "page_count": document.page_count,
            "analyzed_at": datetime.now(timezone.utc).isoformat(),
            "min_text_characters": min_text_characters,
            "ocr_required": bool(ocr_pages),
            "ocr_page_count": len(ocr_pages),
            "ocr_pages": ocr_pages,
            "pages": pages,
        }
    finally:
        document.close()

