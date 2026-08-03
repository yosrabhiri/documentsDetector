"""End-to-end orchestration for the document processing workflow."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Any

from .classifier import classify_ocr_directory
from .identity_detector import auto_detect_identity_cards
from .ocr_processor import run_ocr
from .ocr_refiner import refine_low_confidence_pages
from .pdf_analyzer import analyze_pdf
from .segmenter import build_segments, create_split_pdfs


ProgressCallback = Callable[[str, str], None]


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _timed_step(
    name: str,
    action: Callable[[], Any],
    durations: dict[str, float],
    progress: ProgressCallback | None,
) -> Any:
    if progress:
        progress(name, "started")
    started = perf_counter()
    result = action()
    durations[name] = round(perf_counter() - started, 2)
    if progress:
        progress(name, "completed")
    return result


def process_pdf(
    pdf_path: str | Path,
    *,
    output_root: str | Path = Path("output"),
    dpi: int = 300,
    languages: str = "fra+ara",
    ocr_workers: int = 4,
    cin_workers: int = 2,
    refinement_threshold: float = 0.60,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Run analysis, OCR, classification, CIN detection, refinement and split."""
    total_started = perf_counter()
    durations: dict[str, float] = {}

    analysis = _timed_step(
        "analysis", lambda: analyze_pdf(pdf_path), durations, progress
    )
    reference_dir = Path(output_root).resolve() / analysis["reference"]
    ocr_dir = reference_dir / "ocr"
    documents_dir = reference_dir / "documents"
    reference_dir.mkdir(parents=True, exist_ok=True)
    _write_json(reference_dir / "analysis.json", analysis)

    ocr_report = _timed_step(
        "ocr",
        lambda: run_ocr(
            pdf_path,
            ocr_dir,
            languages=languages,
            dpi=dpi,
            workers=ocr_workers,
            on_page=(
                (lambda page, total: progress("ocr_page", f"{page}/{total}"))
                if progress
                else None
            ),
        ),
        durations,
        progress,
    )
    _write_json(ocr_dir / "ocr.json", ocr_report)

    classification = _timed_step(
        "classification",
        lambda: classify_ocr_directory(ocr_dir),
        durations,
        progress,
    )
    _write_json(ocr_dir / "classification.json", classification)

    cin_detection = _timed_step(
        "cin_detection",
        lambda: auto_detect_identity_cards(
            pdf_path,
            classification,
            reference_dir,
            workers=cin_workers,
        ),
        durations,
        progress,
    )
    _write_json(reference_dir / "cin_detection.json", cin_detection)

    refinement = _timed_step(
        "refinement",
        lambda: refine_low_confidence_pages(
            ocr_dir, threshold=refinement_threshold
        ),
        durations,
        progress,
    )
    _write_json(ocr_dir / "refinement.json", refinement)

    # Refinement updates classification.json, so segmentation must reload it.
    refined_classification = json.loads(
        (ocr_dir / "classification.json").read_text(encoding="utf-8")
    )
    identity_reports = [
        json.loads(
            (reference_dir / f"cin_page_{page_number:04d}.json").read_text(
                encoding="utf-8"
            )
        )
        for page_number in cin_detection["detected_pages"]
    ]
    segmentation = _timed_step(
        "segmentation",
        lambda: build_segments(refined_classification, identity_reports),
        durations,
        progress,
    )
    outputs = _timed_step(
        "pdf_creation",
        lambda: create_split_pdfs(segmentation, documents_dir),
        durations,
        progress,
    )
    segmentation["outputs"] = outputs
    _write_json(reference_dir / "segmentation.json", segmentation)

    reviews = [
        {
            "type": segment["type"],
            "pages": segment["pages"],
            "confidence": segment["confidence"],
        }
        for segment in segmentation["segments"]
        if segment["requires_review"]
    ]
    durations["total"] = round(perf_counter() - total_started, 2)
    final_report = {
        "status": "completed",
        "reference": analysis["reference"],
        "page_count": analysis["page_count"],
        "document_count": len(outputs),
        "review_count": len(reviews),
        "reviews": reviews,
        "configuration": {
            "dpi": dpi,
            "languages": languages,
            "ocr_workers": ocr_workers,
            "cin_workers": cin_workers,
            "refinement_threshold": refinement_threshold,
        },
        "durations_seconds": durations,
        "documents_directory": str(documents_dir),
        "outputs": outputs,
    }
    _write_json(reference_dir / "traitement.json", final_report)
    return final_report
