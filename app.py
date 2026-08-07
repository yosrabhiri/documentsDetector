"""Local Streamlit interface for single and batch PDF processing."""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path
from uuid import uuid4

import streamlit as st

from src.splitdocument.pipeline import process_batch, process_pdf


UPLOAD_DIR = Path("runtime_uploads")
OUTPUT_DIR = Path("output")

STAGE_PROGRESS = {
    "analysis": 0.03,
    "ocr": 0.08,
    "classification": 0.74,
    "cin_detection": 0.80,
    "refinement": 0.88,
    "segmentation": 0.94,
    "pdf_creation": 0.98,
}

STAGE_LABELS = {
    "analysis": "Analyse du PDF",
    "ocr": "Reconnaissance OCR",
    "classification": "Classification des pages",
    "cin_detection": "Détection des CIN",
    "refinement": "Amélioration des pages difficiles",
    "segmentation": "Regroupement des documents",
    "pdf_creation": "Création des fichiers PDF",
}


def safe_pdf_name(filename: str) -> str:
    """Return a filesystem-safe PDF name without path components."""
    name = Path(filename).name
    stem = re.sub(r"[^A-Za-z0-9._ ()-]+", "_", Path(name).stem).strip(" ._")
    return f"{stem or 'document'}.pdf"


def save_uploads(uploaded_files: list) -> tuple[Path, list[Path]]:
    """Save one upload run locally and return its generated paths."""
    run_dir = UPLOAD_DIR / uuid4().hex
    run_dir.mkdir(parents=True, exist_ok=False)
    paths = []
    for uploaded_file in uploaded_files:
        path = run_dir / safe_pdf_name(uploaded_file.name)
        if path in paths:
            raise ValueError(f"Référence dupliquée : {path.stem}")
        path.write_bytes(uploaded_file.getbuffer())
        paths.append(path)
    return run_dir, paths


def remove_upload_run(run_dir: Path) -> None:
    """Remove only temporary uploads created for the completed run."""
    for path in run_dir.glob("*.pdf"):
        path.unlink(missing_ok=True)
    run_dir.rmdir()


def create_documents_archive(report: dict) -> bytes:
    documents_dir = Path(report["documents_directory"])
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for output in report["outputs"]:
            path = documents_dir / output["file"]
            archive.write(path, arcname=path.name)
    return buffer.getvalue()


def create_batch_archive(report: dict) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for result in report["results"]:
            if result["status"] != "completed":
                continue
            documents_dir = Path(result["documents_directory"])
            for output in result["outputs"]:
                path = documents_dir / output["file"]
                archive.write(path, arcname=f"{result['reference']}/{path.name}")
    return buffer.getvalue()


def render_single_results(report: dict) -> None:
    st.subheader("Résultat du traitement")
    metrics = st.columns(4)
    metrics[0].metric("Pages", report["page_count"])
    metrics[1].metric("Documents", report["document_count"])
    metrics[2].metric("À vérifier", report["review_count"])
    metrics[3].metric("Durée", f"{report['durations_seconds']['total']:.0f} s")

    if report["reviews"]:
        st.warning("Une validation est recommandée avant archivage.")
        for review in report["reviews"]:
            pages = ", ".join(str(page) for page in review["pages"])
            st.write(
                f"**{review['type']}** · page(s) {pages} · "
                f"confiance {review['confidence']:.0%}"
            )
    else:
        st.success("Aucun document ne nécessite de validation.")

    rows = [
        {
            "Document": output["type"].replace("_", " ").title(),
            "Pages": ", ".join(str(page) for page in output["pages"]),
            "Fichier": output["file"],
        }
        for output in report["outputs"]
    ]
    st.dataframe(rows, width="stretch", hide_index=True)
    st.download_button(
        "Télécharger tous les PDF",
        data=create_documents_archive(report),
        file_name=f"{report['reference']}_documents.zip",
        mime="application/zip",
        type="primary",
        width="stretch",
        icon=":material/download:",
    )


def render_batch_results(report: dict) -> None:
    st.subheader("Résultat du lot")
    metrics = st.columns(4)
    metrics[0].metric("PDF reçus", report["input_count"])
    metrics[1].metric("Traités", report["completed_count"])
    metrics[2].metric("Échecs", report["failed_count"])
    metrics[3].metric("Durée totale", f"{report['total_duration_seconds']:.0f} s")

    rows = []
    for result in report["results"]:
        rows.append(
            {
                "Référence": result["reference"],
                "Statut": "Terminé" if result["status"] == "completed" else "Échec",
                "Pages": result.get("page_count", ""),
                "Documents": result.get("document_count", ""),
                "À vérifier": result.get("review_count", ""),
                "Durée (s)": result.get("duration_seconds", ""),
            }
        )
    st.dataframe(rows, width="stretch", hide_index=True)

    failures = [result for result in report["results"] if result["status"] == "failed"]
    for failure in failures:
        st.error(f"{failure['reference']} · {failure['error_type']} · {failure['error']}")

    if report["completed_count"]:
        st.download_button(
            "Télécharger le lot traité",
            data=create_batch_archive(report),
            file_name="documents_traites.zip",
            mime="application/zip",
            type="primary",
            width="stretch",
            icon=":material/download:",
        )


def run_single(uploaded_file) -> None:
    run_dir, paths = save_uploads([uploaded_file])
    progress_bar = st.progress(0, text="Préparation")
    status = st.status("Traitement en cours", expanded=True)

    def update_progress(stage: str, state: str) -> None:
        if stage == "ocr_page":
            completed, total = (int(value) for value in state.split("/"))
            progress_bar.progress(
                0.08 + 0.64 * completed / total,
                text=f"OCR · page {completed} sur {total}",
            )
        elif state == "started":
            label = STAGE_LABELS.get(stage, stage)
            status.write(label)
            progress_bar.progress(STAGE_PROGRESS.get(stage, 0), text=label)

    try:
        report = process_pdf(
            paths[0], output_root=OUTPUT_DIR, ocr_workers=4, progress=update_progress
        )
    except Exception as exc:
        progress_bar.empty()
        status.update(label="Échec du traitement", state="error", expanded=True)
        st.error(str(exc))
    else:
        progress_bar.progress(1.0, text="Traitement terminé")
        status.update(label="Traitement terminé", state="complete", expanded=False)
        st.session_state["single_report"] = report
    finally:
        remove_upload_run(run_dir)


def run_batch(uploaded_files: list) -> None:
    run_dir, paths = save_uploads(uploaded_files)
    progress_bar = st.progress(0, text="Préparation du lot")
    status = st.status("Traitement du lot", expanded=True)

    def update_progress(
        index: int, total: int, reference: str, stage: str, state: str
    ) -> None:
        if stage == "ocr_page":
            completed, page_total = (int(value) for value in state.split("/"))
            document_fraction = 0.08 + 0.64 * completed / page_total
            overall = ((index - 1) + document_fraction) / total
            progress_bar.progress(
                overall,
                text=f"{reference} · OCR {completed}/{page_total} · PDF {index}/{total}",
            )
        elif state == "started":
            label = STAGE_LABELS.get(stage, stage)
            overall = ((index - 1) + STAGE_PROGRESS.get(stage, 0)) / total
            progress_bar.progress(overall, text=f"{reference} · {label}")
            status.write(f"{index}/{total} · {reference} · {label}")

    try:
        report = process_batch(
            paths, output_root=OUTPUT_DIR, ocr_workers=4, progress=update_progress
        )
    except Exception as exc:
        progress_bar.empty()
        status.update(label="Échec du lot", state="error", expanded=True)
        st.error(str(exc))
    else:
        progress_bar.progress(1.0, text="Lot terminé")
        final_state = "complete" if report["completed_count"] else "error"
        status.update(label="Traitement du lot terminé", state=final_state, expanded=False)
        st.session_state["batch_report"] = report
    finally:
        remove_upload_run(run_dir)


st.set_page_config(
    page_title="SplitDocument",
    page_icon=":material/document_scanner:",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {max-width: 1120px; padding-top: 2rem; padding-bottom: 3rem;}
    h1, h2, h3 {letter-spacing: 0 !important;}
    [data-testid="stFileUploaderDropzone"] {border-radius: 6px;}
    [data-testid="stMetric"] {
        border: 1px solid #d9dee7; border-radius: 6px;
        padding: 14px 16px; background: #ffffff;
    }
    [data-testid="stMetricLabel"] {color: #465164;}
    </style>
    """,
    unsafe_allow_html=True,
)

header, privacy = st.columns([3, 2], vertical_alignment="bottom")
with header:
    st.title("SplitDocument")
    st.caption("Séparation automatique de dossiers PDF numérisés")
with privacy:
    st.info("Traitement local. Aucun document n’est envoyé vers un service externe.")

workflow = st.segmented_control(
    "Workflow",
    options=["Un PDF", "Traitement par lot"],
    default="Un PDF",
    width="stretch",
)

if workflow == "Un PDF":
    uploaded_file = st.file_uploader("Dossier PDF", type=["pdf"], key="single-pdf")
    if uploaded_file is not None:
        details = st.columns(3)
        details[0].metric("Fichier", safe_pdf_name(uploaded_file.name))
        details[1].metric("Taille", f"{uploaded_file.size / 1024 / 1024:.1f} Mo")
        details[2].metric("OCR", "4 travailleurs")
    if st.button(
        "Séparer le PDF",
        type="primary",
        width="stretch",
        icon=":material/content_cut:",
        disabled=uploaded_file is None,
    ):
        run_single(uploaded_file)
    if "single_report" in st.session_state:
        st.divider()
        render_single_results(st.session_state["single_report"])
else:
    source_mode = st.radio(
        "Source du lot",
        options=["Plusieurs PDF", "Dossier complet"],
        horizontal=True,
    )
    uploaded_files = st.file_uploader(
        "PDF à traiter",
        type=["pdf"],
        accept_multiple_files=True if source_mode == "Plusieurs PDF" else "directory",
        key=f"batch-{source_mode}",
    )
    if uploaded_files:
        total_size = sum(uploaded.size for uploaded in uploaded_files) / 1024 / 1024
        details = st.columns(3)
        details[0].metric("PDF", len(uploaded_files))
        details[1].metric("Taille totale", f"{total_size:.1f} Mo")
        details[2].metric("OCR par PDF", "4 travailleurs")
        st.dataframe(
            [
                {"Référence": Path(safe_pdf_name(uploaded.name)).stem, "Fichier": uploaded.name}
                for uploaded in uploaded_files
            ],
            width="stretch",
            hide_index=True,
        )
    if st.button(
        "Traiter le lot",
        type="primary",
        width="stretch",
        icon=":material/play_arrow:",
        disabled=not uploaded_files,
    ):
        run_batch(uploaded_files)
    if "batch_report" in st.session_state:
        st.divider()
        render_batch_results(st.session_state["batch_report"])
