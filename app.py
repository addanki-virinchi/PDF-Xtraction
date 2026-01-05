"""Flask application for PDF data extraction using QWEN VL model."""

# IMPORTANT: Import config first to set environment variables for temp directories
# before any other modules import torch/transformers
import config  # noqa: F401 - Sets HF_HOME, TEMP, etc. environment variables

import logging
import os
import time
import uuid
import threading
import traceback
from pathlib import Path
from datetime import datetime

from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename

from config import FlaskConfig, ALLOWED_PDF_EXTENSIONS, ALLOWED_EXCEL_EXTENSIONS
from pdf_extractor import extract_text_from_pdf, pdf_to_images, clean_extracted_text
from excel_handler import read_excel_headers, create_output_excel, parse_model_response_to_rows
from prompt_builder import build_vision_messages, build_text_only_messages
from qwen_model import (
    extract_with_vision,
    extract_with_text_only,
    get_model_info,
    is_model_ready,
    preload_model_async,
    get_loading_status
)

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Flask App
# -----------------------------------------------------------------------------
app = Flask(__name__)
app.config.from_object(FlaskConfig)
CORS(app)

# -----------------------------------------------------------------------------
# Jobs store (in-memory) - good for Colab/testing.
# For production, store jobs in Redis/DB.
# -----------------------------------------------------------------------------
JOBS = {}
JOBS_LOCK = threading.Lock()

# How long to keep job metadata + output files (seconds)
JOB_RETENTION_SECONDS = 60 * 60  # 1 hour

def _now_ts() -> float:
    return time.time()

def _job_gc_loop():
    """Garbage collect old jobs + output files."""
    while True:
        try:
            cutoff = _now_ts() - JOB_RETENTION_SECONDS
            to_delete = []
            with JOBS_LOCK:
                for job_id, job in JOBS.items():
                    created_at = job.get("created_at", 0)
                    if created_at < cutoff:
                        to_delete.append(job_id)

            for job_id in to_delete:
                with JOBS_LOCK:
                    job = JOBS.pop(job_id, None)
                if job:
                    out = job.get("output_path")
                    if out:
                        try:
                            Path(out).unlink(missing_ok=True)
                        except Exception:
                            pass
        except Exception:
            logger.exception("Job GC loop error")

        time.sleep(60)

# Start GC thread
threading.Thread(target=_job_gc_loop, daemon=True).start()

# -----------------------------------------------------------------------------
# Model preload guard (prevents double loading)
# -----------------------------------------------------------------------------
_model_preload_started = False
_model_preload_lock = threading.Lock()

def init_model_preload_once():
    """Initialize model preloading on app startup (only once)."""
    global _model_preload_started
    with _model_preload_lock:
        if _model_preload_started:
            return
        _model_preload_started = True
    logger.info("Initiating background model preload...")
    preload_model_async()

# Trigger preload on import (safe because we disable reloader below)
init_model_preload_once()

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def allowed_file(filename: str, allowed_extensions: set) -> bool:
    """Check if file has an allowed extension."""
    return Path(filename).suffix.lower() in allowed_extensions

def generate_output_filename(original_pdf: str) -> str:
    """Generate unique output filename."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = secure_filename(Path(original_pdf).stem)
    if not base_name:
        base_name = "output"
    return f"{base_name}_extracted_{timestamp}.xlsx"

def _set_job(job_id: str, **kwargs):
    with JOBS_LOCK:
        job = JOBS.get(job_id, {})
        job.update(kwargs)
        job["updated_at"] = _now_ts()
        JOBS[job_id] = job

def _get_job(job_id: str):
    with JOBS_LOCK:
        return JOBS.get(job_id)

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "model_info": get_model_info()})

@app.route('/api/model-status', methods=['GET'])
def model_status():
    status = get_loading_status()
    return jsonify({
        "is_ready": is_model_ready(),
        **status,
        "model_name": get_model_info().get("model_name")
    })

@app.route('/api/status/<job_id>', methods=['GET'])
def job_status(job_id):
    job = _get_job(job_id)
    if not job:
        return jsonify({"error": "job not found"}), 404
    # Do not return huge fields by default; status only
    return jsonify({
        "job_id": job_id,
        "status": job.get("status"),
        "progress": job.get("progress"),
        "created_at": job.get("created_at"),
        "updated_at": job.get("updated_at"),
        "error": job.get("error")
    })

@app.route('/api/result/<job_id>', methods=['GET'])
def job_result(job_id):
    job = _get_job(job_id)
    if not job:
        return jsonify({"error": "job not found"}), 404

    status = job.get("status")
    if status == "done":
        return jsonify({
            "success": True,
            "job_id": job_id,
            "fields": job.get("fields"),
            "extracted_data": job.get("extracted_data"),
            "download_url": job.get("download_url"),
        })
    if status == "error":
        return jsonify({
            "success": False,
            "job_id": job_id,
            "error": job.get("error"),
        }), 500

    return jsonify({
        "success": False,
        "job_id": job_id,
        "status": status,
        "progress": job.get("progress"),
    }), 202

@app.route('/api/download/<filename>', methods=['GET'])
def download_file(filename):
    """Download a previously generated output file."""
    file_path = Path(app.config['OUTPUT_FOLDER']) / secure_filename(filename)
    if not file_path.exists():
        return jsonify({"error": "File not found"}), 404
    return send_file(file_path, as_attachment=True)

# -----------------------------------------------------------------------------
# Worker: heavy extraction work runs in background so requests don't timeout
# -----------------------------------------------------------------------------
def _extract_worker(job_id: str, pdf_path: str, excel_path: str, original_pdf_name: str,
                    custom_prompt: str | None, use_vision: bool):
    try:
        _set_job(job_id, status="running", progress="Starting extraction...")

        # Read Excel headers (schema)
        _set_job(job_id, progress="Reading Excel template headers...")
        fields = read_excel_headers(Path(excel_path))
        _set_job(job_id, fields=fields)

        logger.info(f"[{job_id}] Fields to extract: {fields}")

        # Extract text from PDF
        _set_job(job_id, progress="Extracting text from PDF...")
        pdf_text = extract_text_from_pdf(Path(pdf_path))
        pdf_text = clean_extracted_text(pdf_text)

        # Build messages and run extraction
        if use_vision:
            _set_job(job_id, progress="Converting PDF pages to images...")
            images = pdf_to_images(Path(pdf_path))

            _set_job(job_id, progress="Building vision prompt/messages...")
            messages = build_vision_messages(fields, images, custom_prompt, pdf_text)

            _set_job(job_id, progress="Running model inference (vision)...")
            response = extract_with_vision(messages)
        else:
            _set_job(job_id, progress="Building text-only prompt/messages...")
            messages = build_text_only_messages(fields, pdf_text, custom_prompt)

            _set_job(job_id, progress="Running model inference (text-only)...")
            response = extract_with_text_only(messages)

        logger.info(f"[{job_id}] Model response length: {len(response)}")

        _set_job(job_id, progress="Parsing model response...")
        extracted_rows = parse_model_response_to_rows(response, fields)

        # Create output Excel
        _set_job(job_id, progress="Writing output Excel...")
        output_filename = generate_output_filename(original_pdf_name)
        output_path = Path(app.config['OUTPUT_FOLDER']) / output_filename
        create_output_excel(Path(excel_path), output_path, extracted_rows)

        # Cleanup uploaded files (keep output)
        _set_job(job_id, progress="Cleaning up temporary files...")
        Path(pdf_path).unlink(missing_ok=True)
        Path(excel_path).unlink(missing_ok=True)

        _set_job(
            job_id,
            status="done",
            progress="Done",
            extracted_data=extracted_rows,
            output_filename=output_filename,
            output_path=str(output_path),
            download_url=f"/api/download/{output_filename}"
        )

        logger.info(f"[{job_id}] Completed successfully. Output: {output_filename}")

    except Exception:
        err = traceback.format_exc()
        logger.exception(f"[{job_id}] Extraction failed")
        _set_job(job_id, status="error", progress="Error", error=err)

        # Best-effort cleanup
        try:
            Path(pdf_path).unlink(missing_ok=True)
        except Exception:
            pass
        try:
            Path(excel_path).unlink(missing_ok=True)
        except Exception:
            pass

# -----------------------------------------------------------------------------
# Main extraction endpoint (NOW ASYNC)
# -----------------------------------------------------------------------------
@app.route('/api/extract', methods=['POST'])
def extract_data():
    """
    Async extraction endpoint.
    Returns immediately with a job_id, avoiding Cloudflare/browser timeouts.

    Client flow:
      1) POST /api/extract -> {job_id, status_url, result_url}
      2) Poll GET /api/status/<job_id>
      3) GET /api/result/<job_id> when done (contains download_url)
    """
    # Check if model is ready
    loading_status = get_loading_status()
    if loading_status["state"] == "loading":
        elapsed = loading_status.get("elapsed_seconds", 0)
        if loading_status.get("started_at"):
            elapsed = time.time() - loading_status["started_at"]
        return jsonify({
            "error": "Model is still loading. Please wait and try again.",
            "loading_state": loading_status["state"],
            "loading_progress": loading_status["progress"],
            "elapsed_seconds": round(elapsed, 1),
            "retry_after": 30
        }), 503
    elif loading_status["state"] == "error":
        return jsonify({
            "error": f"Model failed to load: {loading_status.get('error')}",
            "loading_state": "error"
        }), 500
    elif loading_status["state"] == "not_started":
        preload_model_async()
        return jsonify({
            "error": "Model loading has been initiated. Please wait and try again.",
            "loading_state": "loading",
            "retry_after": 60
        }), 503

    # Validate files
    if 'pdf_file' not in request.files:
        return jsonify({"error": "No PDF file provided"}), 400
    if 'excel_template' not in request.files:
        return jsonify({"error": "No Excel template provided"}), 400

    pdf_file = request.files['pdf_file']
    excel_file = request.files['excel_template']

    if not pdf_file.filename or not allowed_file(pdf_file.filename, ALLOWED_PDF_EXTENSIONS):
        return jsonify({"error": "Invalid PDF file"}), 400
    if not excel_file.filename or not allowed_file(excel_file.filename, ALLOWED_EXCEL_EXTENSIONS):
        return jsonify({"error": "Invalid Excel file"}), 400

    # Optional parameters
    custom_prompt = request.form.get('custom_prompt', '').strip() or None
    use_vision = request.form.get('use_vision', 'true').lower() == 'true'

    # Save uploaded files
    session_id = str(uuid.uuid4())[:8]
    pdf_filename = secure_filename(f"{session_id}_{pdf_file.filename}")
    excel_filename = secure_filename(f"{session_id}_{excel_file.filename}")

    pdf_path = Path(app.config['UPLOAD_FOLDER']) / pdf_filename
    excel_path = Path(app.config['UPLOAD_FOLDER']) / excel_filename

    pdf_file.save(pdf_path)
    excel_file.save(excel_path)

    job_id = str(uuid.uuid4())

    # Create job entry
    _set_job(
        job_id,
        status="queued",
        progress="Queued",
        created_at=_now_ts(),
        pdf_filename=pdf_filename,
        excel_filename=excel_filename
    )

    logger.info(f"[{job_id}] Queued: PDF={pdf_filename}, Template={excel_filename}")

    # Start background worker
    t = threading.Thread(
        target=_extract_worker,
        args=(job_id, str(pdf_path), str(excel_path), pdf_file.filename, custom_prompt, use_vision),
        daemon=True
    )
    t.start()

    # Return immediately => prevents "context canceled" / 524 timeouts
    return jsonify({
        "job_id": job_id,
        "status_url": f"/api/status/{job_id}",
        "result_url": f"/api/result/{job_id}"
    }), 202


# -----------------------------------------------------------------------------
# Main entry
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    # IMPORTANT:
    # debug=False and use_reloader=False prevents double model loading (OOM)
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
