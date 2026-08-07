import json
from pathlib import Path

import fitz
import pytest

import src.splitdocument.pipeline as pipeline_module
from src.splitdocument.classifier import classify_text
from src.splitdocument.identity_detector import (
    _face_evidence,
    select_identity_candidate_pages,
)
from src.splitdocument.ocr_processor import OcrError, parse_page_selection, run_ocr
from src.splitdocument.ocr_refiner import select_refinement_pages
from src.splitdocument.pdf_analyzer import PdfAnalysisError, analyze_pdf
from src.splitdocument.segmenter import build_segments, create_split_pdfs
from src.splitdocument.validation import (
    evaluate_segmentation,
    generate_validation_scenarios,
)


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


def test_rejects_invalid_ocr_worker_count(tmp_path: Path) -> None:
    with pytest.raises(OcrError, match="workers must be at least 1"):
        run_ocr(tmp_path / "missing.pdf", tmp_path / "output", workers=0)


def test_parallel_ocr_preserves_page_order(tmp_path: Path) -> None:
    pdf_path = tmp_path / "ORDER.pdf"
    document = fitz.open()
    for number in range(1, 4):
        page = document.new_page()
        page.insert_text(
            (72, 72),
            f"Searchable content for page number {number} with enough characters.",
        )
    document.save(pdf_path)
    document.close()

    report = run_ocr(pdf_path, tmp_path / "ocr", workers=2)

    assert report["workers"] == 2
    assert [page["page_number"] for page in report["pages"]] == [1, 2, 3]
    assert all(page["method"] == "embedded_text" for page in report["pages"])


def test_process_pdf_orchestrates_complete_workflow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "REF.pdf"
    source.write_bytes(b"%PDF-test")
    calls = []
    classification = {
        "source_file": str(source),
        "reference": "REF",
        "classified_page_count": 1,
        "pages": [
            {
                "page_number": 1,
                "type": "contrat",
                "confidence": 0.9,
                "matched_keywords": [],
                "candidates": [],
            }
        ],
    }

    monkeypatch.setattr(
        pipeline_module,
        "analyze_pdf",
        lambda path: {
            "source_file": str(source),
            "reference": "REF",
            "page_count": 1,
        },
    )

    def fake_ocr(*args, **kwargs):
        calls.append("ocr")
        return {
            "source_file": str(source),
            "reference": "REF",
            "processed_page_count": 1,
            "pages": [],
        }

    monkeypatch.setattr(pipeline_module, "run_ocr", fake_ocr)
    monkeypatch.setattr(
        pipeline_module,
        "classify_ocr_directory",
        lambda path: calls.append("classification") or classification,
    )
    monkeypatch.setattr(
        pipeline_module,
        "auto_detect_identity_cards",
        lambda *args, **kwargs: calls.append("cin_detection")
        or {
            "analyzed_page_count": 0,
            "detected_cin_count": 0,
            "detected_pages": [],
            "workers": 2,
        },
    )
    monkeypatch.setattr(
        pipeline_module,
        "refine_low_confidence_pages",
        lambda *args, **kwargs: calls.append("refinement")
        or {"candidate_count": 0, "accepted_count": 0, "pages": []},
    )
    monkeypatch.setattr(
        pipeline_module,
        "build_segments",
        lambda *args: calls.append("segmentation")
        or {
            "source_file": str(source),
            "reference": "REF",
            "page_count": 1,
            "segment_count": 1,
            "segments": [
                {
                    "type": "contrat",
                    "pages": [1],
                    "confidence": 0.9,
                    "requires_review": False,
                }
            ],
        },
    )
    monkeypatch.setattr(
        pipeline_module,
        "create_split_pdfs",
        lambda *args: calls.append("pdf_creation")
        or [{"type": "contrat", "pages": [1], "file": "contrat_REF.pdf"}],
    )

    report = pipeline_module.process_pdf(source, output_root=tmp_path / "output")

    assert calls == [
        "ocr",
        "classification",
        "cin_detection",
        "refinement",
        "segmentation",
        "pdf_creation",
    ]
    assert report["status"] == "completed"
    assert report["document_count"] == 1
    assert (tmp_path / "output" / "REF" / "traitement.json").is_file()


def test_process_batch_keeps_successes_when_one_pdf_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "REF001.pdf"
    second = tmp_path / "REF002.pdf"
    first.write_bytes(b"pdf")
    second.write_bytes(b"pdf")

    def fake_process(path, **kwargs):
        reference = Path(path).stem
        if reference == "REF002":
            raise OcrError("OCR failed")
        return {
            "reference": reference,
            "page_count": 3,
            "document_count": 2,
            "review_count": 0,
            "durations_seconds": {"total": 1.2},
            "documents_directory": str(tmp_path / reference),
            "outputs": [],
        }

    monkeypatch.setattr(pipeline_module, "process_pdf", fake_process)
    report = pipeline_module.process_batch(
        [first, second], output_root=tmp_path / "output"
    )

    assert report["status"] == "partial"
    assert report["completed_count"] == 1
    assert report["failed_count"] == 1
    assert report["results"][0]["reference"] == "REF001"
    assert report["results"][1]["error_type"] == "OcrError"


def test_process_batch_rejects_duplicate_references(tmp_path: Path) -> None:
    first = tmp_path / "a" / "REF.pdf"
    second = tmp_path / "b" / "REF.pdf"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_bytes(b"pdf")
    second.write_bytes(b"pdf")

    with pytest.raises(ValueError, match="Duplicate PDF references"):
        pipeline_module.process_batch([first, second], output_root=tmp_path / "out")


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


def test_evaluation_detects_merged_same_type_documents() -> None:
    expected = {
        "reference": "REF",
        "segments": [
            {"type": "contrat", "pages": [1, 2]},
            {"type": "contrat", "pages": [3, 4]},
            {"type": "statuts", "pages": [5]},
        ],
    }
    actual = {
        "segments": [
            {"type": "contrat", "pages": [1, 2, 3, 4]},
            {"type": "statuts", "pages": [5]},
        ]
    }

    result = evaluate_segmentation(expected, actual)

    assert result["page_type_accuracy"] == 1.0
    assert result["exact_segment_rate"] < 1.0
    assert result["boundary_recall"] == 0.5


def test_generates_five_validation_scenarios(tmp_path: Path) -> None:
    source_path = tmp_path / "SOURCE.pdf"
    document = fitz.open()
    for number in range(3):
        page = document.new_page()
        page.insert_text((72, 72), f"Document page {number + 1}")
    document.save(source_path)
    document.close()
    segmentation_path = tmp_path / "segmentation.json"
    segmentation_path.write_text(
        json.dumps(
            {
                "segments": [
                    {"type": "contrat", "pages": [1]},
                    {"type": "statuts", "pages": [2]},
                    {"type": "cin", "pages": [3]},
                ]
            }
        ),
        encoding="utf-8",
    )

    report = generate_validation_scenarios(
        source_path, segmentation_path, tmp_path / "scenarios"
    )

    assert report["scenario_count"] == 5
    assert all(Path(item["pdf"]).is_file() for item in report["scenarios"])
    repeated = json.loads(
        (tmp_path / "scenarios" / "scenario_type_repete.expected.json").read_text()
    )
    assert repeated["segments"][0]["type"] == "contrat"
    assert repeated["segments"][1]["type"] == "contrat"
