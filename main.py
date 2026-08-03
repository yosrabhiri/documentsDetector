"""Command-line entry point for PDF analysis and OCR."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.splitdocument.classifier import classify_ocr_directory
from src.splitdocument.identity_detector import (
    auto_detect_identity_cards,
    detect_identity_card,
)
from src.splitdocument.ocr_processor import OcrError, parse_page_selection, run_ocr
from src.splitdocument.ocr_refiner import refine_low_confidence_pages
from src.splitdocument.pdf_analyzer import PdfAnalysisError, analyze_pdf
from src.splitdocument.pipeline import process_pdf
from src.splitdocument.segmenter import build_segments, create_split_pdfs, load_identity_reports


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze scanned PDFs and extract text.")
    commands = parser.add_subparsers(dest="command", required=True)

    analyze_command = commands.add_parser("analyze", help="Identify pages requiring OCR")
    analyze_command.add_argument("pdf", type=Path, help="Path to the PDF to analyze")
    analyze_command.add_argument("--output", type=Path, help="JSON report path")
    analyze_command.add_argument("--min-text-characters", type=int, default=20)

    ocr_command = commands.add_parser("ocr", help="Extract French and Arabic text")
    ocr_command.add_argument("pdf", type=Path, help="Path to the PDF to process")
    ocr_command.add_argument("--pages", help="Page selection, for example: 1,3-5")
    ocr_command.add_argument("--dpi", type=int, default=300)
    ocr_command.add_argument("--languages", default="fra+ara")
    ocr_command.add_argument("--workers", type=int, default=4)

    classify_command = commands.add_parser("classify", help="Classify OCR page texts")
    classify_command.add_argument("ocr_dir", type=Path, help="Directory containing ocr.json")

    identity_command = commands.add_parser("detect-cin", help="Detect CIN faces on a page")
    identity_command.add_argument("pdf", type=Path)
    identity_command.add_argument("--page", type=int, required=True)

    auto_identity_command = commands.add_parser(
        "auto-detect-cin", help="Detect CIN pages among unknown classifications"
    )
    auto_identity_command.add_argument("pdf", type=Path)
    auto_identity_command.add_argument("ocr_dir", type=Path)
    auto_identity_command.add_argument("--workers", type=int, default=2)

    split_command = commands.add_parser("split", help="Group pages and create separate PDFs")
    split_command.add_argument("ocr_dir", type=Path, help="Directory containing classification.json")

    refine_command = commands.add_parser("refine", help="Retry low-confidence OCR pages")
    refine_command.add_argument("ocr_dir", type=Path)
    refine_command.add_argument("--threshold", type=float, default=0.60)

    process_command = commands.add_parser(
        "process", help="Run the complete PDF processing workflow"
    )
    process_command.add_argument("pdf", type=Path)
    process_command.add_argument("--output-root", type=Path, default=Path("output"))
    process_command.add_argument("--dpi", type=int, default=300)
    process_command.add_argument("--languages", default="fra+ara")
    process_command.add_argument("--ocr-workers", type=int, default=4)
    process_command.add_argument("--cin-workers", type=int, default=2)
    process_command.add_argument("--refinement-threshold", type=float, default=0.60)
    return parser


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "process":
            labels = {
                "analysis": "Analysis",
                "ocr": "OCR",
                "classification": "Classification",
                "cin_detection": "CIN detection",
                "refinement": "OCR refinement",
                "segmentation": "Segmentation",
                "pdf_creation": "PDF creation",
            }

            def show_progress(stage: str, status: str) -> None:
                if stage == "ocr_page":
                    print(f"  OCR page {status}", flush=True)
                    return
                print(f"[{labels.get(stage, stage)}] {status}", flush=True)

            report = process_pdf(
                args.pdf,
                output_root=args.output_root,
                dpi=args.dpi,
                languages=args.languages,
                ocr_workers=args.ocr_workers,
                cin_workers=args.cin_workers,
                refinement_threshold=args.refinement_threshold,
                progress=show_progress,
            )
            report_path = (
                args.output_root.resolve()
                / report["reference"]
                / "traitement.json"
            )
            print(f"Pages processed: {report['page_count']}")
            print(f"PDF files created: {report['document_count']}")
            print(f"Documents requiring review: {report['review_count']}")
            print(f"Total time: {report['durations_seconds']['total']:.2f}s")
            print(f"Final report: {report_path}")
            return 0

        if args.command == "auto-detect-cin":
            classification_path = args.ocr_dir / "classification.json"
            classification = json.loads(
                classification_path.read_text(encoding="utf-8")
            )
            report = auto_detect_identity_cards(
                args.pdf,
                classification,
                args.ocr_dir.parent,
                workers=args.workers,
            )
            report_path = args.ocr_dir.parent / "cin_detection.json"
            write_json(report_path, report)
            print(f"Unknown pages analyzed: {report['analyzed_page_count']}")
            print(f"CIN pages detected: {report['detected_cin_count']}")
            print(f"CIN detection report: {report_path.resolve()}")
            return 0

        if args.command == "refine":
            report = refine_low_confidence_pages(args.ocr_dir, threshold=args.threshold)
            report_path = args.ocr_dir / "refinement.json"
            write_json(report_path, report)
            print(f"Candidates: {report['candidate_count']}")
            print(f"Improvements accepted: {report['accepted_count']}")
            print(f"Refinement report: {report_path.resolve()}")
            return 0

        if args.command == "split":
            classification_path = args.ocr_dir / "classification.json"
            classification = json.loads(classification_path.read_text(encoding="utf-8"))
            reference_dir = args.ocr_dir.parent
            identities = load_identity_reports(reference_dir)
            report = build_segments(classification, identities)
            documents_dir = reference_dir / "documents"
            report["outputs"] = create_split_pdfs(report, documents_dir)
            report_path = reference_dir / "segmentation.json"
            write_json(report_path, report)
            review_count = sum(segment["requires_review"] for segment in report["segments"])
            print(f"Segments: {report['segment_count']}")
            print(f"PDF files created: {len(report['outputs'])}")
            print(f"Segments requiring review: {review_count}")
            print(f"Segmentation report: {report_path.resolve()}")
            return 0

        if args.command == "detect-cin":
            report = detect_identity_card(args.pdf, args.page)
            reference = args.pdf.stem
            report_path = Path("output") / reference / f"cin_page_{args.page:04d}.json"
            write_json(report_path, report)
            print(f"Type: {report['type']}")
            print(f"Layout: {report['layout']}")
            print(f"Confidence: {report['confidence']:.2f}")
            print(f"CIN report: {report_path.resolve()}")
            return 0

        if args.command == "classify":
            report = classify_ocr_directory(args.ocr_dir)
            report_path = args.ocr_dir / "classification.json"
            write_json(report_path, report)
            known = sum(page["type"] != "inconnu" for page in report["pages"])
            print(f"Classified pages: {known}/{report['classified_page_count']}")
            print(f"Classification report: {report_path.resolve()}")
            return 0

        if args.command == "ocr":
            preliminary = analyze_pdf(args.pdf)
            selected_pages = parse_page_selection(args.pages, preliminary["page_count"])
            output_dir = Path("output") / preliminary["reference"] / "ocr"
            report = run_ocr(
                args.pdf,
                output_dir,
                pages=selected_pages,
                languages=args.languages,
                dpi=args.dpi,
                workers=args.workers,
                on_page=lambda page, total: print(
                    f"OCR page {page} ({total} selected)...", flush=True
                ),
            )
            report_path = output_dir / "ocr.json"
            write_json(report_path, report)
            print(f"Processed pages: {report['processed_page_count']}")
            print(f"OCR report: {report_path.resolve()}")
            return 0

        report = analyze_pdf(args.pdf, min_text_characters=args.min_text_characters)
        output_path = args.output or (
            Path("output") / report["reference"] / "analysis.json"
        )
        write_json(output_path, report)
        print(f"Reference: {report['reference']}")
        print(f"Pages: {report['page_count']}")
        print(f"Pages requiring OCR: {report['ocr_page_count']}")
        print(f"Report: {output_path.resolve()}")
        return 0
    except (PdfAnalysisError, OcrError, ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
