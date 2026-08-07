"""Tools for analyzing and splitting scanned PDF documents."""

from .pdf_analyzer import analyze_pdf
from .ocr_processor import run_ocr
from .classifier import classify_ocr_directory, classify_text
from .identity_detector import auto_detect_identity_cards, detect_identity_card
from .segmenter import build_segments, create_split_pdfs
from .ocr_refiner import refine_low_confidence_pages
from .pipeline import process_batch, process_pdf
from .validation import evaluate_segmentation, generate_validation_scenarios

__all__ = [
    "analyze_pdf", "run_ocr", "classify_ocr_directory", "classify_text",
    "detect_identity_card",
    "auto_detect_identity_cards",
    "build_segments", "create_split_pdfs",
    "refine_low_confidence_pages",
    "process_pdf",
    "process_batch",
    "generate_validation_scenarios", "evaluate_segmentation",
]
