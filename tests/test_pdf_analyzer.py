from pathlib import Path

import fitz
import pytest

from src.splitdocument.classifier import classify_text
from src.splitdocument.identity_detector import (
    _face_evidence,
    select_identity_candidate_pages,
)
from src.splitdocument.ocr_processor import OcrError, parse_page_selection
from src.splitdocument.ocr_refiner import select_refinement_pages
from src.splitdocument.pdf_analyzer import PdfAnalysisError, analyze_pdf
from src.splitdocument.segmenter import build_segments, create_split_pdfs


def test_detects_text_page_and_scanned_page(tmp_path: Path) -> None:
    pdf_path = tmp_path / "REF-001.pdf"
    document = fitz.open()
    text_page = document.new_page()
    text_page.insert_text((72, 72), "This page contains enough searchable text.")
    document.new_page()
    document.save(pdf_path)
    document.close()

    report = analyze_pdf(pdf_path)

    assert report["reference"] == "REF-001"
    assert report["page_count"] == 2
    assert report["ocr_pages"] == [2]
    assert report["pages"][0]["ocr_required"] is False
    assert report["pages"][1]["ocr_required"] is True


def test_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(PdfAnalysisError, match="not found"):
        analyze_pdf(tmp_path / "missing.pdf")


def test_rejects_negative_threshold(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="zero or greater"):
        analyze_pdf(tmp_path / "anything.pdf", min_text_characters=-1)


def test_parses_page_selection() -> None:
    assert parse_page_selection("1,3-5,3", 6) == [1, 3, 4, 5]


def test_rejects_page_outside_document() -> None:
    with pytest.raises(OcrError, match="outside document range"):
        parse_page_selection("1,8", 6)


@pytest.mark.parametrize(
    ("text", "expected_type"),
    [
        ("CARTE D'IDENTIFICATION FISCALE - Matricule fiscal", "carte_identification_fiscale"),
        ("مضمون من السجل التجاري شخص معنوي", "registre_commerce"),
        ("STATUTS Entre les soussignés", "statuts"),
        ("Texte sans titre suffisamment distinctif", "inconnu"),
    ],
)
def test_classifies_bilingual_text(text: str, expected_type: str) -> None:
    assert classify_text(text)["type"] == expected_type


def test_recognizes_cin_recto_from_tolerant_ocr() -> None:
    side, confidence, evidence = _face_evidence(
        "بطافه التميف الولية 12345678", has_barcode=False
    )
    assert side == "recto"
    assert confidence >= 0.5
    assert "titre_recto_partiel" in evidence


def test_recognizes_cin_verso_from_barcode() -> None:
    side, confidence, evidence = _face_evidence("العنوان تونس", has_barcode=True)
    assert side == "verso"
    assert confidence >= 0.45
    assert "code_barres" in evidence


def test_groups_only_unknown_pages_bounded_by_same_type() -> None:
    classification = {
        "source_file": "sample.pdf",
        "reference": "REF",
        "pages": [
            {"page_number": 1, "type": "statuts", "confidence": 0.9},
            {"page_number": 2, "type": "inconnu", "confidence": 0.0},
            {"page_number": 3, "type": "statuts", "confidence": 0.8},
            {"page_number": 4, "type": "inconnu", "confidence": 0.0},
            {"page_number": 5, "type": "contrat_abonnement", "confidence": 0.9},
        ],
    }
    report = build_segments(classification)
    assert report["segments"][0]["type"] == "statuts"
    assert report["segments"][0]["pages"] == [1, 2, 3]
    assert report["segments"][1]["type"] == "inconnu"


def test_identity_report_overrides_unknown_page() -> None:
    classification = {
        "source_file": "sample.pdf",
        "reference": "REF",
        "pages": [{"page_number": 1, "type": "inconnu", "confidence": 0.0}],
    }
    identities = [{"page_number": 1, "type": "carte_identite", "confidence": 0.8}]
    report = build_segments(classification, identities)
    assert report["segments"][0]["type"] == "carte_identite"
    assert report["segments"][0]["requires_review"] is False


def test_flags_low_confidence_segment_for_review() -> None:
    classification = {
        "source_file": "sample.pdf",
        "reference": "REF",
        "pages": [{"page_number": 1, "type": "carte_fiscale", "confidence": 0.55}],
    }
    assert build_segments(classification)["segments"][0]["requires_review"] is True


def test_selects_only_known_low_confidence_pages_for_refinement() -> None:
    report = {
        "pages": [
            {"page_number": 1, "type": "carte_fiscale", "confidence": 0.55},
            {"page_number": 2, "type": "inconnu", "confidence": 0.0},
            {"page_number": 3, "type": "contrat", "confidence": 0.9},
        ]
    }
    assert select_refinement_pages(report) == [1]


def test_selects_only_unknown_pages_for_visual_identity_detection() -> None:
    report = {
        "pages": [
            {"page_number": 1, "type": "inconnu", "confidence": 0.0},
            {"page_number": 2, "type": "contrat", "confidence": 0.9},
            {"page_number": 3, "type": "inconnu", "confidence": 0.0},
        ]
    }
    assert select_identity_candidate_pages(report) == [1, 3]


def test_creates_a_pdf_for_each_known_segment(tmp_path: Path) -> None:
    source_path = tmp_path / "REF.pdf"
    source = fitz.open()
    source.new_page()
    source.new_page()
    source.save(source_path)
    source.close()
    report = {
        "source_file": str(source_path),
        "reference": "REF",
        "segments": [
            {"type": "contrat", "pages": [1]},
            {"type": "inconnu", "pages": [2]},
        ],
    }
    outputs = create_split_pdfs(report, tmp_path / "documents")
    assert len(outputs) == 1
    assert fitz.open(tmp_path / "documents" / "contrat_REF.pdf").page_count == 1
