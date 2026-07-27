"""Retry OCR on low-confidence pages with lightweight image cleanup."""

from __future__ import annotations

import io
import json
import subprocess
from pathlib import Path
from typing import Any

import fitz
from PIL import Image, ImageFilter, ImageOps

from .classifier import classify_text
from .ocr_processor import DEFAULT_TESSERACT_PATH, OcrError


def select_refinement_pages(
    classification_report: dict[str, Any], threshold: float = 0.60
) -> list[int]:
    """Select known text classifications below the confidence threshold."""
    return [
        page["page_number"]
        for page in classification_report["pages"]
        if page["type"] != "inconnu" and page["confidence"] < threshold
    ]


def _preprocess(image: Image.Image) -> Image.Image:
    grayscale = ImageOps.grayscale(image)
    contrasted = ImageOps.autocontrast(grayscale, cutoff=1)
    return contrasted.filter(ImageFilter.MedianFilter(3))


def _tesseract(image: Image.Image, tessdata: Path, executable: Path) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    command = [
        str(executable), "stdin", "stdout", "--tessdata-dir", str(tessdata),
        "-l", "fra+ara", "--psm", "3",
    ]
    try:
        result = subprocess.run(
            command, input=buffer.getvalue(), capture_output=True, check=True, timeout=180
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise OcrError("Refined OCR failed") from exc
    return result.stdout.decode("utf-8", errors="replace").strip()


def refine_low_confidence_pages(
    ocr_dir: str | Path,
    *,
    threshold: float = 0.60,
    tessdata_dir: str | Path = Path("models/tessdata"),
    tesseract_path: str | Path = DEFAULT_TESSERACT_PATH,
) -> dict[str, Any]:
    """Retry selected pages and keep only same-type confidence improvements."""
    directory = Path(ocr_dir).resolve()
    classification_path = directory / "classification.json"
    ocr_path = directory / "ocr.json"
    classification = json.loads(classification_path.read_text(encoding="utf-8"))
    ocr_report = json.loads(ocr_path.read_text(encoding="utf-8"))
    candidates = select_refinement_pages(classification, threshold)
    classification_by_page = {
        page["page_number"]: page for page in classification["pages"]
    }
    ocr_by_page = {page["page_number"]: page for page in ocr_report["pages"]}

    document = fitz.open(ocr_report["source_file"])
    results = []
    try:
        for page_number in candidates:
            current = classification_by_page[page_number]
            pixmap = document[page_number - 1].get_pixmap(
                matrix=fitz.Matrix(300 / 72, 300 / 72),
                colorspace=fitz.csRGB,
                alpha=False,
            )
            image = Image.open(io.BytesIO(pixmap.tobytes("png")))
            refined_text = _tesseract(
                _preprocess(image), Path(tessdata_dir).resolve(), Path(tesseract_path).resolve()
            )
            refined = classify_text(refined_text)
            accepted = (
                refined["type"] == current["type"]
                and refined["confidence"] > current["confidence"]
            )
            result = {
                "page_number": page_number,
                "type": current["type"],
                "original_confidence": current["confidence"],
                "refined_confidence": refined["confidence"],
                "accepted": accepted,
            }
            results.append(result)
            if not accepted:
                continue

            text_path = directory / ocr_by_page[page_number]["text_file"]
            text_path.write_text(refined_text, encoding="utf-8")
            current.update(
                confidence=refined["confidence"],
                matched_keywords=refined["matched_keywords"],
                candidates=refined["candidates"],
                refinement="median_filter",
                original_confidence=result["original_confidence"],
            )
            ocr_by_page[page_number].update(
                character_count=len(refined_text),
                preview=" ".join(refined_text.split())[:200],
                refinement="median_filter",
            )
    finally:
        document.close()

    classification_path.write_text(
        json.dumps(classification, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    ocr_path.write_text(json.dumps(ocr_report, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "threshold": threshold,
        "candidate_count": len(candidates),
        "accepted_count": sum(result["accepted"] for result in results),
        "pages": results,
    }
