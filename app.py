"""Local Streamlit interface for the SplitDocument processing pipeline."""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import streamlit as st

from src.splitdocument.pipeline import process_pdf


UPLOAD_DIR = Path("runtime_uploads")
OUTPUT_DIR = Path("output")

STAGE_PROGRESS = {
    "analysis": 5,
    "ocr": 10,
    "classification": 74,
    "cin_detection": 80,
    "refinement": 88,
    "segmentation": 94,
    "pdf_creation": 98,
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
    """Return a filesystem-safe PDF name without leaking path components."""
    name = Path(filename).name
    stem = re.sub(r"[^A-Za-z0-9._ ()-]+", "_", Path(name).stem).strip(" ._")
    return f"{stem or 'document'}.pdf"


def create_documents_archive(report: dict) -> bytes:
    """Build an in-memory ZIP containing only the final split PDFs."""
    documents_dir = Path(report["documents_directory"])
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for output in report["outputs"]:
            path = documents_dir / output["file"]
            archive.write(path, arcname=path.name)
    return buffer.getvalue()


def render_results(report: dict) -> None:
    st.subheader("Résultat du traitement")
    metrics = st.columns(4)
    metrics[0].metric("Pages", report["page_count"])
    metrics[1].metric("Documents", report["document_count"])
    metrics[2].metric("À vérifier", report["review_count"])
    metrics[3].metric(
        "Durée",
        f"{report['durations_seconds']['total']:.0f} s",
    )

    if report["reviews"]:
        with st.container(border=True):
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

    archive = create_documents_archive(report)
    st.download_button(
        "Télécharger tous les PDF",
        data=archive,
        file_name=f"{report['reference']}_documents.zip",
        mime="application/zip",
        type="primary",
        width="stretch",
        icon=":material/download:",
    )

    with st.expander("Téléchargements individuels"):
        documents_dir = Path(report["documents_directory"])
        for output in report["outputs"]:
            path = documents_dir / output["file"]
            left, right = st.columns([4, 1])
            left.write(
                f"**{output['type'].replace('_', ' ').title()}**  \n"
                f"Pages {', '.join(str(page) for page in output['pages'])}"
            )
            right.download_button(
                "PDF",
                data=path.read_bytes(),
                file_name=path.name,
                mime="application/pdf",
                key=f"download-{path.name}",
                width="stretch",
                icon=":material/download:",
            )


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
        border: 1px solid #d9dee7;
        border-radius: 6px;
        padding: 14px 16px;
        background: #ffffff;
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

uploaded_file = st.file_uploader(
    "Dossier PDF",
    type=["pdf"],
    accept_multiple_files=False,
)

if uploaded_file is not None:
    file_size_mb = uploaded_file.size / (1024 * 1024)
    details = st.columns(3)
    details[0].metric("Fichier", safe_pdf_name(uploaded_file.name))
    details[1].metric("Taille", f"{file_size_mb:.1f} Mo")
    details[2].metric("Langues OCR", "Français + arabe")

start_processing = st.button(
    "Séparer le PDF",
    type="primary",
    width="stretch",
    icon=":material/content_cut:",
    disabled=uploaded_file is None,
)

if start_processing and uploaded_file is not None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    input_path = UPLOAD_DIR / safe_pdf_name(uploaded_file.name)
    input_path.write_bytes(uploaded_file.getbuffer())

    progress_bar = st.progress(0, text="Préparation")
    status = st.status("Traitement en cours", expanded=True)

    def update_progress(stage: str, state: str) -> None:
        if stage == "ocr_page":
            completed, total = (int(value) for value in state.split("/"))
            percent = 10 + round(62 * completed / total)
            progress_bar.progress(
                percent,
                text=f"OCR · page {completed} sur {total}",
            )
            return
        label = STAGE_LABELS.get(stage, stage)
        percent = STAGE_PROGRESS.get(stage, 0)
        if state == "started":
            status.write(label)
            progress_bar.progress(percent, text=label)

    try:
            report = process_pdf(
                input_path,
                output_root=OUTPUT_DIR,
                ocr_workers=4,
                progress=update_progress,
            )
    except Exception as exc:
        progress_bar.empty()
        status.update(label="Échec du traitement", state="error", expanded=True)
        st.error(str(exc))
    else:
        progress_bar.progress(100, text="Traitement terminé")
        status.update(label="Traitement terminé", state="complete", expanded=False)
        st.session_state["last_report"] = report

if "last_report" in st.session_state:
    st.divider()
    render_results(st.session_state["last_report"])
