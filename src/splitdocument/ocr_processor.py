"""Extract text from PDF pages with local Tesseract OCR."""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import fitz

from .pdf_analyzer import analyze_pdf


DEFAULT_DPI = 300
DEFAULT_LANGUAGES = "fra+ara"
DEFAULT_TESSERACT_PATH = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")


class OcrError(RuntimeError):
    """Raised when the OCR engine cannot process a document."""


def parse_page_selection(selection: str | None, page_count: int) -> list[int]:
    """Parse a selection such as ``1,3-5`` into sorted page numbers."""
    if not selection:
        return list(range(1, page_count + 1))

    pages: set[int] = set()
    try:
        for part in selection.split(","):
            limits = [int(value.strip()) for value in part.split("-")]
            if len(limits) == 1:
                pages.add(limits[0])
            elif len(limits) == 2 and limits[0] <= limits[1]:
                pages.update(range(limits[0], limits[1] + 1))
            else:
                raise ValueError
    except ValueError as exc:
        raise OcrError(f"Invalid page selection: {selection}") from exc

    invalid = sorted(page for page in pages if page < 1 or page > page_count)
    if invalid:
        raise OcrError(f"Pages outside document range: {invalid}")
    return sorted(pages)


def run_ocr(
    pdf_path: str | Path,
    output_dir: str | Path,
    *,
    pages: Iterable[int] | None = None,
    languages: str = DEFAULT_LANGUAGES,
    dpi: int = DEFAULT_DPI,
    tesseract_path: str | Path = DEFAULT_TESSERACT_PATH,
    tessdata_dir: str | Path = Path("models/tessdata"),
    on_page: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """OCR selected pages and save one UTF-8 text file per page."""
    if dpi < 72:
        raise OcrError("DPI must be at least 72")

    tesseract = Path(tesseract_path).resolve()
    tessdata = Path(tessdata_dir).resolve()
    destination = Path(output_dir).resolve()
    if not tesseract.is_file():
        raise OcrError(f"Tesseract executable not found: {tesseract}")
    if not tessdata.is_dir():
        raise OcrError(f"Tessdata directory not found: {tessdata}")

    missing_languages = [
        language
        for language in languages.split("+")
        if not (tessdata / f"{language}.traineddata").is_file()
    ]
    if missing_languages:
        raise OcrError(f"Missing language models: {', '.join(missing_languages)}")

    analysis = analyze_pdf(pdf_path)
    selected_pages = sorted(set(pages or range(1, analysis["page_count"] + 1)))
    invalid = [page for page in selected_pages if page < 1 or page > analysis["page_count"]]
    if invalid:
        raise OcrError(f"Pages outside document range: {invalid}")

    destination.mkdir(parents=True, exist_ok=True)
    page_reports = []
    document = fitz.open(analysis["source_file"])
    try:
        for page_number in selected_pages:
            if on_page:
                on_page(page_number, len(selected_pages))

            page = document[page_number - 1]
            page_analysis = analysis["pages"][page_number - 1]
            if page_analysis["ocr_required"]:
                scale = dpi / 72
                pixmap = page.get_pixmap(
                    matrix=fitz.Matrix(scale, scale),
                    colorspace=fitz.csRGB,
                    alpha=False,
                )
                command = [
                    str(tesseract), "stdin", "stdout",
                    "--tessdata-dir", str(tessdata),
                    "-l", languages, "--psm", "3",
                ]
                try:
                    result = subprocess.run(
                        command,
                        input=pixmap.tobytes("png"),
                        capture_output=True,
                        check=True,
                        timeout=180,
                    )
                except subprocess.TimeoutExpired as exc:
                    raise OcrError(f"OCR timed out on page {page_number}") from exc
                except subprocess.CalledProcessError as exc:
                    error = exc.stderr.decode("utf-8", errors="replace").strip()
                    raise OcrError(f"OCR failed on page {page_number}: {error}") from exc
                text = result.stdout.decode("utf-8", errors="replace").strip()
                method = "tesseract"
            else:
                text = page.get_text("text").strip()
                method = "embedded_text"

            text_file = destination / f"page_{page_number:04d}.txt"
            text_file.write_text(text, encoding="utf-8")
            page_reports.append(
                {
                    "page_number": page_number,
                    "method": method,
                    "languages": languages if method == "tesseract" else None,
                    "character_count": len(text),
                    "text_file": text_file.name,
                    "preview": " ".join(text.split())[:200],
                }
            )
    finally:
        document.close()

    return {
        "source_file": analysis["source_file"],
        "reference": analysis["reference"],
        "languages": languages,
        "dpi": dpi,
        "processed_page_count": len(page_reports),
        "pages": page_reports,
    }
