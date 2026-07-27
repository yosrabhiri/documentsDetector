"""Group classified pages into documents and write separate PDF files."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import fitz


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"Report not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def build_segments(
    classification_report: dict[str, Any],
    identity_reports: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Resolve safe continuations and form consecutive document segments."""
    pages = [
        {
            "page_number": page["page_number"],
            "type": page["type"],
            "confidence": page["confidence"],
            "source": "text_classification",
        }
        for page in classification_report["pages"]
    ]
    pages.sort(key=lambda page: page["page_number"])

    by_number = {page["page_number"]: page for page in pages}
    for identity in identity_reports or []:
        if identity.get("type") != "carte_identite":
            continue
        page = by_number.get(identity["page_number"])
        if page:
            page.update(
                type="carte_identite",
                confidence=identity["confidence"],
                source="identity_detection",
                identity_layout=identity.get("layout"),
            )

    # Fill only unknown runs enclosed by the same known type. This does not
    # assume any global ordering of document categories.
    index = 0
    while index < len(pages):
        if pages[index]["type"] != "inconnu":
            index += 1
            continue
        start = index
        while index < len(pages) and pages[index]["type"] == "inconnu":
            index += 1
        previous_type = pages[start - 1]["type"] if start > 0 else None
        next_type = pages[index]["type"] if index < len(pages) else None
        if previous_type and previous_type == next_type:
            for page in pages[start:index]:
                page.update(type=previous_type, confidence=0.4, source="bounded_continuation")

    segments = []
    current = None
    for page in pages:
        page_type = page["type"]
        if current is None or current["type"] != page_type:
            current = {
                "type": page_type,
                "pages": [page["page_number"]],
                "page_sources": [page["source"]],
                "confidence_values": [page["confidence"]],
            }
            segments.append(current)
        else:
            current["pages"].append(page["page_number"])
            current["page_sources"].append(page["source"])
            current["confidence_values"].append(page["confidence"])

    for segment in segments:
        segment["start_page"] = segment["pages"][0]
        segment["end_page"] = segment["pages"][-1]
        segment["confidence"] = round(
            sum(segment.pop("confidence_values")) / len(segment["pages"]), 2
        )
        segment["requires_review"] = (
            segment["type"] == "inconnu" or segment["confidence"] < 0.60
        )

    return {
        "source_file": classification_report["source_file"],
        "reference": classification_report["reference"],
        "page_count": len(pages),
        "segment_count": len(segments),
        "segments": segments,
    }


def create_split_pdfs(
    segmentation_report: dict[str, Any], output_dir: str | Path
) -> list[dict[str, Any]]:
    """Create one PDF for each known segment, preserving original pages."""
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    source = fitz.open(segmentation_report["source_file"])
    known_segments = [
        segment for segment in segmentation_report["segments"]
        if segment["type"] != "inconnu"
    ]
    totals = Counter(segment["type"] for segment in known_segments)
    occurrences: Counter[str] = Counter()
    outputs = []
    try:
        for segment in known_segments:
            document_type = segment["type"]
            occurrences[document_type] += 1
            suffix = f"_{occurrences[document_type]:02d}" if totals[document_type] > 1 else ""
            filename = f"{document_type}_{segmentation_report['reference']}{suffix}.pdf"
            output_path = destination / filename
            extracted = fitz.open()
            try:
                for page_number in segment["pages"]:
                    extracted.insert_pdf(source, from_page=page_number - 1, to_page=page_number - 1)
                extracted.save(output_path, garbage=4, deflate=True)
            finally:
                extracted.close()
            outputs.append(
                {
                    "type": document_type,
                    "pages": segment["pages"],
                    "file": filename,
                }
            )
    finally:
        source.close()
    return outputs


def load_identity_reports(directory: str | Path) -> list[dict[str, Any]]:
    """Read sanitized CIN reports generated beside the document output."""
    return [_load_json(path) for path in sorted(Path(directory).glob("cin_page_*.json"))]
