"""Explainable page classification based on bilingual OCR text."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any


MIN_CONFIDENCE = 0.45

# Strong titles receive more weight than supporting vocabulary.
CLASS_RULES: dict[str, tuple[tuple[str, float], ...]] = {
    "contrat_abonnement": (
        ("contrat d abonnement", 0.75),
        ("termes et conditions", 0.55),
        ("amadeus", 0.20),
    ),
    "annexe_contrat": (
        ("annexe 1", 0.75),
        ("contrat d abonnement", 0.35),
    ),
    "avenant": (("avenant", 0.85), ("ajout", 0.20)),
    "fiche_signaletique": (
        ("fiche signaletique", 0.85),
        ("type de licence", 0.25),
    ),
    "demande_licence": (
        ("demande de licence", 0.85),
        ("demande licence", 0.85),
        ("directeur general", 0.20),
    ),
    "statuts": (
        ("statuts", 0.85),
        ("entre les soussignes", 0.25),
        ("forme denomination siege", 0.20),
        ("formalites de depots et de publicite", 0.60),
        ("gerant associe", 0.25),
    ),
    "cahier_charges": (
        ("cahier des charges relatif", 0.80),
        ("agence de voyages de categorie", 0.65),
        ("cahier des charges", 0.20),
        ("agence de voyages", 0.20),
        ("office national du tourisme", 0.20),
    ),
    "carte_identification_fiscale": (
        ("carte d identification fiscale", 0.90),
        ("بطاقة التعريف الجبائية", 0.90),
        ("matricule fiscal", 0.30),
        ("مراقبة الاداءات", 0.25),
    ),
    "registre_national_entreprises": (
        ("registre national des entreprises", 0.85),
        ("rne public", 0.60),
        ("السجل الوطني للمؤسسات", 0.85),
    ),
    "registre_commerce": (
        ("registre du commerce", 0.80),
        ("registre de commerce", 0.80),
        ("السجل التجاري", 0.90),
        ("مضمون من السجل", 0.75),
    ),
    "carte_identite": (
        ("carte d identite nationale", 0.90),
        ("بطاقة التعريف الوطنية", 0.90),
        ("الجمهورية التونسية", 0.15),
    ),
}


def normalize_text(text: str) -> str:
    """Normalize OCR text while preserving Arabic letters."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"[^\w\u0600-\u06ff]+", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def classify_text(text: str, *, min_confidence: float = MIN_CONFIDENCE) -> dict[str, Any]:
    """Classify OCR text and expose all matching evidence."""
    normalized = normalize_text(text)
    candidates = []
    for document_type, rules in CLASS_RULES.items():
        matches = []
        score = 0.0
        for keyword, weight in rules:
            normalized_keyword = normalize_text(keyword)
            if normalized_keyword in normalized:
                matches.append(keyword)
                score += weight
        if matches:
            candidates.append(
                {
                    "type": document_type,
                    "confidence": round(min(score, 1.0), 2),
                    "matched_keywords": matches,
                }
            )

    candidates.sort(key=lambda item: item["confidence"], reverse=True)
    best = candidates[0] if candidates else None
    if best is None or best["confidence"] < min_confidence:
        return {"type": "inconnu", "confidence": 0.0, "matched_keywords": [], "candidates": candidates}
    return {**best, "candidates": candidates}


def classify_ocr_directory(ocr_dir: str | Path) -> dict[str, Any]:
    """Classify every page listed in an OCR report."""
    directory = Path(ocr_dir).resolve()
    report_path = directory / "ocr.json"
    if not report_path.is_file():
        raise ValueError(f"OCR report not found: {report_path}")
    ocr_report = json.loads(report_path.read_text(encoding="utf-8"))

    pages = []
    for page in ocr_report["pages"]:
        text_path = directory / page["text_file"]
        prediction = classify_text(text_path.read_text(encoding="utf-8"))
        pages.append({"page_number": page["page_number"], **prediction})

    return {
        "source_file": ocr_report["source_file"],
        "reference": ocr_report["reference"],
        "classified_page_count": len(pages),
        "pages": pages,
    }
