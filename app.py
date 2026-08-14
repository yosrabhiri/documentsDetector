"""Local Streamlit interface for single and batch PDF processing."""

from __future__ import annotations

import io
import html
import re
import shutil
import time
import zipfile
from collections import Counter
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


def format_size(size_bytes: int) -> str:
    """Format a byte count for compact interface tables."""
    size = float(size_bytes)
    for unit in ("o", "Ko", "Mo", "Go"):
        if size < 1024 or unit == "Go":
            return f"{size:.0f} {unit}" if unit in {"o", "Ko"} else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} Go"


def upload_signature(uploaded_files: list) -> tuple[tuple[str, int], ...]:
    """Identify an upload selection without reading its contents again."""
    return tuple(sorted((safe_pdf_name(item.name), item.size) for item in uploaded_files))


def duplicate_references(uploaded_files: list) -> list[str]:
    """Return normalized references occurring more than once in an upload."""
    references = [Path(safe_pdf_name(item.name)).stem for item in uploaded_files]
    counts = Counter(references)
    return sorted(reference for reference, count in counts.items() if count > 1)


def clear_stale_result(state_key: str, signature_key: str, uploaded_files: list) -> None:
    """Hide a previous result as soon as the selected input changes."""
    current = upload_signature(uploaded_files)
    previous = st.session_state.get(signature_key)
    if previous is not None and previous != current:
        st.session_state.pop(state_key, None)
    st.session_state[signature_key] = current


def upload_rows(uploaded_files: list) -> list[dict[str, str]]:
    return [
        {
            "Référence": Path(safe_pdf_name(uploaded.name)).stem,
            "Fichier": uploaded.name,
            "Taille": format_size(uploaded.size),
        }
        for uploaded in uploaded_files
    ]


def render_table(rows: list[dict], *, empty_message: str = "Aucune donnée") -> None:
    """Render a small safe HTML table without importing Pandas."""
    if not rows:
        st.caption(empty_message)
        return

    columns = list(rows[0])
    header = "".join(f"<th>{html.escape(str(column))}</th>" for column in columns)
    body = "".join(
        "<tr>"
        + "".join(
            f"<td>{html.escape(str(row.get(column, '')))}</td>" for column in columns
        )
        + "</tr>"
        for row in rows
    )
    st.markdown(
        f'<div class="sd-table-wrap"><table class="sd-table">'
        f"<thead><tr>{header}</tr></thead><tbody>{body}</tbody></table></div>",
        unsafe_allow_html=True,
    )


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
    """Best-effort cleanup that never hides a completed processing result."""
    for attempt in range(3):
        try:
            shutil.rmtree(run_dir)
            return
        except FileNotFoundError:
            return
        except PermissionError:
            if attempt < 2:
                time.sleep(0.25 * (attempt + 1))

    # OneDrive or an antivirus can briefly retain a handle on an uploaded PDF.
    # The ignored runtime directory can safely be cleaned during a later run.


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
    title, state = st.columns([4, 1], vertical_alignment="center")
    title.subheader(f"Résultat · {report['reference']}")
    state.success("Terminé", icon=":material/check_circle:")
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
    render_table(rows)
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
    title, state = st.columns([4, 1], vertical_alignment="center")
    title.subheader("Résultat du lot")
    if report["failed_count"]:
        state.warning("Partiel", icon=":material/warning:")
    else:
        state.success("Terminé", icon=":material/check_circle:")
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
    render_table(rows)

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
    run_dir = None
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
        run_dir, paths = save_uploads([uploaded_file])
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
        if run_dir is not None:
            remove_upload_run(run_dir)


def run_batch(uploaded_files: list) -> None:
    run_dir = None
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
        run_dir, paths = save_uploads(uploaded_files)
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
        if run_dir is not None:
            remove_upload_run(run_dir)


st.set_page_config(
    page_title="SplitDocument",
    page_icon=":material/document_scanner:",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #17212b;
        --muted: #647180;
        --line: #dce2e7;
        --surface: #ffffff;
        --canvas: #f5f7f8;
        --teal: #087f7b;
        --teal-soft: #e7f5f3;
        --green: #24744f;
        --green-soft: #edf7f1;
    }
    .stApp {background: var(--canvas); color: var(--ink);}
    [data-testid="stHeader"] {background: rgba(245, 247, 248, .92);}
    .block-container {max-width: 1180px; padding-top: 1.25rem; padding-bottom: 3rem;}
    h1, h2, h3, p {letter-spacing: 0 !important;}
    h2 {font-size: 1.18rem !important; color: var(--ink);}
    .sd-header {
        display: flex; align-items: center; justify-content: space-between;
        gap: 24px; padding: 12px 0 24px; border-bottom: 1px solid var(--line);
        margin-bottom: 24px;
    }
    .sd-brand {display: flex; align-items: center; gap: 14px; min-width: 0;}
    .sd-mark {
        width: 44px; height: 44px; display: grid; place-items: center;
        border-radius: 6px; background: var(--teal); color: white;
        font: 700 21px/1 Arial, sans-serif; flex: 0 0 auto;
        box-shadow: 0 6px 16px rgba(8, 127, 123, .18);
    }
    .sd-title {font-size: 1.72rem; line-height: 1.1; font-weight: 720; color: var(--ink);}
    .sd-subtitle {font-size: .91rem; color: var(--muted); margin-top: 5px;}
    .sd-privacy {
        display: inline-flex; align-items: center; gap: 8px; flex: 0 0 auto;
        color: var(--green); background: var(--green-soft);
        border: 1px solid #c9e5d4; border-radius: 999px;
        padding: 8px 12px; font-size: .84rem; font-weight: 650;
    }
    .sd-dot {width: 7px; height: 7px; border-radius: 50%; background: var(--green);}
    [data-testid="stFileUploaderDropzone"] {
        border-radius: 6px; min-height: 156px; border: 1.5px dashed #aab7c2;
        background: var(--surface); transition: border-color .15s, background .15s;
    }
    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: var(--teal); background: #f8fcfb;
    }
    [data-testid="stFileUploaderDropzone"] button {
        border-color: #aab7c2; background: white;
    }
    [data-testid="stMetric"] {
        border: 1px solid #d9dee7; border-radius: 6px;
        padding: 14px 16px; background: var(--surface);
        box-shadow: 0 1px 2px rgba(23, 33, 43, .03);
    }
    [data-testid="stMetricLabel"] {color: var(--muted); font-size: .79rem;}
    [data-testid="stMetricValue"] {color: var(--ink); font-size: 1.42rem;}
    [data-testid="stAlert"] {border-radius: 6px;}
    div[data-testid="stSegmentedControl"] {
        max-width: 520px; background: #e9edef; padding: 4px;
        border-radius: 6px; margin-bottom: 8px;
    }
    div[data-testid="stSegmentedControl"] button {min-height: 38px; border-radius: 4px;}
    .sd-table-wrap {
        width: 100%; overflow-x: auto; border: 1px solid var(--line);
        border-radius: 6px; background: var(--surface); margin: 10px 0;
    }
    .sd-table {width: 100%; border-collapse: collapse; font-size: .88rem;}
    .sd-table th {
        padding: 11px 14px; text-align: left; color: var(--muted);
        background: #f8fafb; border-bottom: 1px solid var(--line);
        font-size: .76rem; font-weight: 700; text-transform: uppercase;
    }
    .sd-table td {
        padding: 12px 14px; color: var(--ink); border-bottom: 1px solid #edf0f2;
    }
    .sd-table tr:last-child td {border-bottom: 0;}
    .sd-table tbody tr:hover {background: #f8fcfb;}
    }
    .stButton > button, .stDownloadButton > button {
        min-height: 2.9rem; border-radius: 5px; font-weight: 650;
    }
    .stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {
        box-shadow: 0 4px 12px rgba(8, 127, 123, .16);
    }
    hr {border-color: var(--line) !important;}
    small, [data-testid="stCaptionContainer"] {color: var(--muted) !important;}
    @media (max-width: 700px) {
        .block-container {padding: 1rem 1rem 2rem;}
        .sd-header {align-items: flex-start; flex-direction: column; gap: 14px;}
        .sd-title {font-size: 1.45rem;}
        .sd-privacy {font-size: .78rem;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="sd-header">
        <div class="sd-brand">
            <div class="sd-mark">S</div>
            <div>
                <div class="sd-title">SplitDocument</div>
                <div class="sd-subtitle">Traitement et séparation de dossiers PDF numérisés</div>
            </div>
        </div>
        <div class="sd-privacy"><span class="sd-dot"></span>Traitement local et confidentiel</div>
    </div>
    """,
    unsafe_allow_html=True,
)

workflow = st.segmented_control(
    "Workflow",
    options=["Un PDF", "Traitement par lot"],
    default="Un PDF",
    width="stretch",
)

if workflow == "Un PDF":
    st.subheader("Document à traiter")
    uploaded_file = st.file_uploader(
        "Dossier PDF", type=["pdf"], key="single-pdf", label_visibility="collapsed"
    )
    if uploaded_file is not None:
        clear_stale_result(
            "single_report", "single_upload_signature", [uploaded_file]
        )
        render_table(upload_rows([uploaded_file]))
        st.caption("OCR français + arabe · 4 workers")
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
    st.subheader("Documents à traiter")
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
    duplicates = duplicate_references(uploaded_files) if uploaded_files else []
    if uploaded_files:
        clear_stale_result(
            "batch_report", "batch_upload_signature", uploaded_files
        )
        total_size = sum(uploaded.size for uploaded in uploaded_files)
        details = st.columns(3)
        details[0].metric("PDF", len(uploaded_files))
        details[1].metric("Taille totale", format_size(total_size))
        details[2].metric("Références", len({Path(safe_pdf_name(item.name)).stem for item in uploaded_files}))
        render_table(upload_rows(uploaded_files))
        if duplicates:
            st.warning(
                "Impossible d’effectuer cette opération : deux fichiers ont le même nom. "
                "Supprimez l’un des fichiers en double pour continuer.",
                icon=":material/content_copy:",
            )
        st.caption("Traitement indépendant par référence · OCR français + arabe · 4 workers")
    if st.button(
        "Traiter le lot",
        type="primary",
        width="stretch",
        icon=":material/play_arrow:",
        disabled=not uploaded_files or bool(duplicates),
    ):
        run_batch(uploaded_files)
    if "batch_report" in st.session_state:
        st.divider()
        render_batch_results(st.session_state["batch_report"])
