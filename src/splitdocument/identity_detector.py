"""Locate Tunisian identity-card faces and identify recto/verso layouts."""

from __future__ import annotations

import io
import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import fitz
from PIL import Image

from .classifier import normalize_text
from .ocr_processor import DEFAULT_TESSERACT_PATH, OcrError


RECTO_TERMS = (
    "بطاقة التعريف الوطنية",
    "الجمهورية التونسية",
    "تاريخ الولادة",
    "carte nationale d identite",
)
VERSO_TERMS = ("اسم ولقب الام", "المهنة", "العنوان", "تونس في")


def _groups(values: list[int], threshold: int, max_gap: int = 8) -> list[tuple[int, int]]:
    active = [index for index, value in enumerate(values) if value >= threshold]
    if not active:
        return []
    result = []
    start = previous = active[0]
    for index in active[1:]:
        if index - previous > max_gap:
            result.append((start, previous + 1))
            start = index
        previous = index
    result.append((start, previous + 1))
    return result


def locate_card_regions(image: Image.Image) -> list[tuple[int, int, int, int]]:
    """Find one or two card-like regions using low-resolution ink projections."""
    scale = min(1.0, 700 / image.width)
    preview = image.resize((round(image.width * scale), round(image.height * scale)))
    ink = preview.convert("L").point(lambda pixel: 255 if pixel < 215 else 0, mode="L")
    # BOX resizing to one pixel computes the mean ink density for each axis.
    y_projection = list(ink.resize((1, preview.height), Image.Resampling.BOX).getdata())
    row_groups = _groups(y_projection, 3, max_gap=12)
    row_groups = [
        group for group in row_groups
        if group[1] - group[0] >= preview.height * 0.06
        and group[1] - group[0] <= preview.height * 0.55
    ]
    if not row_groups:
        return []

    # Identity cards usually form the densest compact horizontal band on an A4 scan.
    y1, y2 = max(row_groups, key=lambda group: sum(y_projection[group[0]:group[1]]))
    band = ink.crop((0, y1, preview.width, y2))
    band_x = list(band.resize((preview.width, 1), Image.Resampling.BOX).getdata())
    column_groups = _groups(band_x, 4, max_gap=18)
    column_groups = [group for group in column_groups if group[1] - group[0] >= preview.width * 0.12]

    if len(column_groups) >= 2:
        candidates = column_groups[:2]
    elif column_groups:
        x1, x2 = column_groups[0]
        width = x2 - x1
        height = y2 - y1
        if width / max(height, 1) > 2.4:
            middle = (x1 + x2) // 2
            candidates = [(x1, middle), (middle, x2)]
        else:
            candidates = [(x1, x2)]
    else:
        return []

    inverse = 1 / scale
    margin = round(8 * inverse)
    regions = []
    for x1, x2 in candidates:
        box = (
            max(0, round(x1 * inverse) - margin),
            max(0, round(y1 * inverse) - margin),
            min(image.width, round(x2 * inverse) + margin),
            min(image.height, round(y2 * inverse) + margin),
        )
        width, height = box[2] - box[0], box[3] - box[1]
        ratio = width / max(height, 1)
        if 1.15 <= ratio <= 2.25:
            regions.append(box)
    return regions


def _ocr_crop(image: Image.Image, tessdata_dir: Path, tesseract_path: Path) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    command = [
        str(tesseract_path), "stdin", "stdout", "--tessdata-dir", str(tessdata_dir),
        "-l", "ara+fra", "--psm", "6",
    ]
    try:
        result = subprocess.run(command, input=buffer.getvalue(), capture_output=True, check=True, timeout=90)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise OcrError("Identity-card OCR failed") from exc
    return result.stdout.decode("utf-8", errors="replace").strip()


def _has_barcode(image: Image.Image) -> bool:
    """Detect the dense vertical transitions typical of the old CIN barcode."""
    sample = image.convert("L").resize((500, 300)).crop((0, 165, 500, 300))
    column_means = list(sample.resize((500, 1), Image.Resampling.BOX).getdata())
    dark = [value < 180 for value in column_means]
    transitions = sum(left != right for left, right in zip(dark, dark[1:]))
    return transitions >= 40 and sum(dark) >= 35


def _face_evidence(text: str, *, has_barcode: bool) -> tuple[str, float, list[str]]:
    normalized = normalize_text(text)
    recto = [term for term in RECTO_TERMS if normalize_text(term) in normalized]
    verso = [term for term in VERSO_TERMS if normalize_text(term) in normalized]
    has_id_number = bool(re.search(r"(?<!\d)\d{8}(?!\d)", normalized))
    compact = normalized.replace(" ", "")
    fuzzy_recto_title = "بطا" in compact and ("الوط" in compact or "الول" in compact)
    recto_score = min(
        1.0,
        len(recto) * 0.25 + (0.45 if fuzzy_recto_title and not recto else 0)
        + (0.15 if has_id_number else 0),
    )
    verso_score = min(
        1.0,
        len(verso) * 0.22 + (0.45 if has_barcode else 0)
        + (0.12 if has_id_number else 0),
    )
    if max(recto_score, verso_score) < 0.35:
        return "unknown", round(max(recto_score, verso_score), 2), recto + verso
    if recto_score >= verso_score:
        evidence = recto + (["titre_recto_partiel"] if fuzzy_recto_title and not recto else [])
        return "recto", round(recto_score, 2), evidence + (["numero_8_chiffres"] if has_id_number else [])
    evidence = verso + (["code_barres"] if has_barcode else [])
    return "verso", round(verso_score, 2), evidence + (["numero_8_chiffres"] if has_id_number else [])


def detect_identity_card(
    pdf_path: str | Path,
    page_number: int,
    *,
    tessdata_dir: str | Path = Path("models/tessdata"),
    tesseract_path: str | Path = DEFAULT_TESSERACT_PATH,
) -> dict[str, Any]:
    """Detect CIN faces on one PDF page without returning personal OCR text."""
    document = fitz.open(Path(pdf_path).resolve())
    try:
        if page_number < 1 or page_number > document.page_count:
            raise ValueError(f"Page outside document range: {page_number}")
        pixmap = document[page_number - 1].get_pixmap(
            matrix=fitz.Matrix(300 / 72, 300 / 72), colorspace=fitz.csRGB, alpha=False
        )
        image = Image.open(io.BytesIO(pixmap.tobytes("png")))
    finally:
        document.close()

    regions = locate_card_regions(image)
    faces = []
    for region in regions:
        crop = image.crop(region)
        text = _ocr_crop(crop, Path(tessdata_dir).resolve(), Path(tesseract_path).resolve())
        side, confidence, evidence = _face_evidence(text, has_barcode=_has_barcode(crop))
        faces.append({"side": side, "confidence": confidence, "evidence": evidence})

    recognized = [face for face in faces if face["side"] != "unknown"]
    sides = {face["side"] for face in recognized}
    if {"recto", "verso"}.issubset(sides):
        layout = "recto_verso_meme_page"
    elif len(recognized) == 1:
        layout = recognized[0]["side"]
    else:
        layout = "indetermine"
    confidence = round(sum(face["confidence"] for face in recognized) / max(len(recognized), 1), 2)
    return {
        "page_number": page_number,
        "type": "carte_identite" if recognized else "inconnu",
        "layout": layout,
        "confidence": confidence,
        "detected_regions": len(regions),
        "faces": faces,
    }


def select_identity_candidate_pages(
    classification_report: dict[str, Any],
) -> list[int]:
    """Return pages for which text classification could not determine a type."""
    return [
        page["page_number"]
        for page in classification_report["pages"]
        if page["type"] == "inconnu"
    ]


def auto_detect_identity_cards(
    pdf_path: str | Path,
    classification_report: dict[str, Any],
    output_dir: str | Path,
    *,
    workers: int = 2,
) -> dict[str, Any]:
    """Run visual CIN detection on unknown pages and save sanitized positives."""
    if workers < 1:
        raise ValueError("workers must be at least 1")

    candidates = select_identity_candidate_pages(classification_report)
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)

    def analyze(page_number: int) -> dict[str, Any]:
        return detect_identity_card(pdf_path, page_number)

    with ThreadPoolExecutor(max_workers=min(workers, max(len(candidates), 1))) as executor:
        results = list(executor.map(analyze, candidates))

    detected = []
    for result in results:
        if result["type"] != "carte_identite":
            continue
        report_path = destination / f"cin_page_{result['page_number']:04d}.json"
        report_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        detected.append(result)

    return {
        "analyzed_page_count": len(candidates),
        "detected_cin_count": len(detected),
        "detected_pages": [result["page_number"] for result in detected],
        "workers": workers,
    }
