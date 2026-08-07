"""Generate local validation scenarios and score segmentation reports."""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import fitz


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _scenario_segments(segmentation_report: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"type": segment["type"], "source_pages": list(segment["pages"])}
        for segment in segmentation_report["segments"]
        if segment["type"] != "inconnu"
    ]


def _create_scenario_pdf(
    source: fitz.Document,
    destination: Path,
    specifications: list[dict[str, Any]],
    *,
    rotate_first_page: bool = False,
) -> dict[str, Any]:
    generated = fitz.open()
    expected_segments = []
    try:
        for specification in specifications:
            output_pages = []
            for position, source_page in enumerate(specification["source_pages"]):
                generated.insert_pdf(
                    source, from_page=source_page - 1, to_page=source_page - 1
                )
                generated_page = generated.page_count
                output_pages.append(generated_page)
                if rotate_first_page and position == 0:
                    page = generated[generated_page - 1]
                    page.set_rotation((page.rotation + 90) % 360)
            expected_segments.append(
                {
                    "type": specification["type"],
                    "pages": output_pages,
                    "source_pages": specification["source_pages"],
                }
            )
        generated.save(destination, garbage=4, deflate=True)
    finally:
        generated.close()
    return {
        "reference": destination.stem,
        "page_count": sum(len(segment["pages"]) for segment in expected_segments),
        "segments": expected_segments,
    }


def generate_validation_scenarios(
    source_pdf: str | Path,
    segmentation_report: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Create deterministic PDFs that exercise ordering and boundary behavior."""
    source_path = Path(source_pdf).resolve()
    report = json.loads(Path(segmentation_report).read_text(encoding="utf-8"))
    canonical = _scenario_segments(report)
    if len(canonical) < 2:
        raise ValueError("At least two known segments are required")

    shuffled = canonical.copy()
    random.Random(42).shuffle(shuffled)
    scenarios = {
        "scenario_ordre_inverse": (list(reversed(canonical)), False),
        "scenario_documents_absents": (canonical[::2], False),
        "scenario_type_repete": ([canonical[0], canonical[0], *canonical[1:]], False),
        "scenario_ordre_aleatoire": (shuffled, False),
        "scenario_pages_tournees": (canonical, True),
    }

    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    index = {
        "source_file": str(source_path),
        "scenario_count": len(scenarios),
        "scenarios": [],
    }
    source = fitz.open(source_path)
    try:
        for name, (specifications, rotate) in scenarios.items():
            pdf_path = destination / f"{name}.pdf"
            expected = _create_scenario_pdf(
                source, pdf_path, specifications, rotate_first_page=rotate
            )
            expected_path = destination / f"{name}.expected.json"
            _write_json(expected_path, expected)
            index["scenarios"].append(
                {
                    "name": name,
                    "pdf": str(pdf_path),
                    "expected": str(expected_path),
                    "page_count": expected["page_count"],
                }
            )
    finally:
        source.close()
    _write_json(destination / "scenarios.json", index)
    return index


def evaluate_segmentation(
    expected_report: dict[str, Any], actual_report: dict[str, Any]
) -> dict[str, Any]:
    """Compare page labels, boundaries and exact document segments."""
    expected_segments = expected_report["segments"]
    actual_segments = actual_report["segments"]

    def page_labels(segments: list[dict[str, Any]]) -> dict[int, str]:
        return {
            page: segment["type"]
            for segment in segments
            for page in segment["pages"]
        }

    expected_labels = page_labels(expected_segments)
    actual_labels = page_labels(actual_segments)
    expected_pages = set(expected_labels)
    actual_pages = set(actual_labels)
    correct_labels = sum(
        actual_labels.get(page) == expected_labels[page] for page in expected_pages
    )

    expected_boundaries = {
        segment["pages"][0] for segment in expected_segments[1:] if segment["pages"]
    }
    actual_boundaries = {
        segment["pages"][0] for segment in actual_segments[1:] if segment["pages"]
    }
    boundary_matches = len(expected_boundaries & actual_boundaries)
    boundary_precision = (
        boundary_matches / len(actual_boundaries) if actual_boundaries else 1.0
    )
    boundary_recall = (
        boundary_matches / len(expected_boundaries) if expected_boundaries else 1.0
    )
    boundary_f1 = (
        2 * boundary_precision * boundary_recall / (boundary_precision + boundary_recall)
        if boundary_precision + boundary_recall
        else 0.0
    )

    expected_exact = Counter(
        (segment["type"], tuple(segment["pages"])) for segment in expected_segments
    )
    actual_exact = Counter(
        (segment["type"], tuple(segment["pages"])) for segment in actual_segments
    )
    exact_matches = sum((expected_exact & actual_exact).values())
    return {
        "reference": expected_report["reference"],
        "expected_page_count": len(expected_pages),
        "actual_page_count": len(actual_pages),
        "missing_pages": sorted(expected_pages - actual_pages),
        "extra_pages": sorted(actual_pages - expected_pages),
        "page_type_accuracy": round(correct_labels / max(len(expected_pages), 1), 4),
        "expected_segment_count": len(expected_segments),
        "actual_segment_count": len(actual_segments),
        "exact_segment_count": exact_matches,
        "exact_segment_rate": round(exact_matches / max(len(expected_segments), 1), 4),
        "boundary_precision": round(boundary_precision, 4),
        "boundary_recall": round(boundary_recall, 4),
        "boundary_f1": round(boundary_f1, 4),
    }


def evaluate_report_files(
    expected_path: str | Path,
    actual_path: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    expected = json.loads(Path(expected_path).read_text(encoding="utf-8"))
    actual = json.loads(Path(actual_path).read_text(encoding="utf-8"))
    result = evaluate_segmentation(expected, actual)
    if output_path:
        _write_json(Path(output_path), result)
    return result
