"""Flask application for PDF data extraction using QWEN VL model."""

# IMPORTANT: Import config first to set environment variables for temp directories
# before any other modules import torch/transformers
import config  # noqa: F401 - Sets HF_HOME, TEMP, etc. environment variables

import logging
import uuid
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(FlaskConfig)
CORS(app)

# Start background model preloading when app starts
# This ensures the model is ready before the first request
def init_model_preload():
    """Initialize model preloading on app startup."""
    logger.info("Initiating background model preload...")
    preload_model_async()

# Trigger preload (will run when module is imported)
init_model_preload()


def allowed_file(filename: str, allowed_extensions: set) -> bool:
    """Check if file has an allowed extension."""
    return Path(filename).suffix.lower() in allowed_extensions


def generate_output_filename(original_pdf: str) -> str:
    """Generate unique output filename."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = Path(original_pdf).stem
    return f"{base_name}_extracted_{timestamp}.xlsx"


@app.route('/')
def index():
    """Render the main upload page."""
    return render_template('index.html')


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "model_info": get_model_info()})


@app.route('/api/model-status', methods=['GET'])
def model_status():
    """
    Get detailed model loading status.

    Use this endpoint to poll for model readiness before submitting extraction requests.

    Returns:
        - loading_state: "not_started", "loading", "ready", or "error"
        - loading_progress: Human-readable progress message
        - is_ready: Boolean indicating if model is ready for inference
        - elapsed_seconds: Time since loading started (if loading)
        - load_time_seconds: Total time to load (if complete)
    """
    status = get_loading_status()
    return jsonify({
        "is_ready": is_model_ready(),
        **status,
        "model_name": get_model_info().get("model_name")
    })


@app.route('/api/extract', methods=['POST'])
def extract_data():
    """
    Main extraction endpoint.

    Accepts:
        - pdf_file: PDF document to extract from
        - excel_template: Excel file defining the schema/fields
        - custom_prompt: (optional) Additional extraction instructions
        - use_vision: (optional) Whether to use vision mode (default: true)

    Returns:
        - Extracted data as JSON or downloadable Excel file

    Note:
        If the model is still loading, returns 503 with loading status.
        Poll /api/model-status to check when model is ready.
    """
    try:
        # Check if model is ready
        loading_status = get_loading_status()
        if loading_status["state"] == "loading":
            elapsed = loading_status.get("elapsed_seconds", 0)
            if loading_status["started_at"]:
                import time
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
            # Trigger loading if not started
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
        
        # Get optional parameters
        custom_prompt = request.form.get('custom_prompt', '').strip() or None
        use_vision = request.form.get('use_vision', 'true').lower() == 'true'
        return_json = request.form.get('return_json', 'false').lower() == 'true'
        
        # Save uploaded files
        session_id = str(uuid.uuid4())[:8]
        pdf_filename = secure_filename(f"{session_id}_{pdf_file.filename}")
        excel_filename = secure_filename(f"{session_id}_{excel_file.filename}")
        
        pdf_path = Path(app.config['UPLOAD_FOLDER']) / pdf_filename
        excel_path = Path(app.config['UPLOAD_FOLDER']) / excel_filename
        
        pdf_file.save(pdf_path)
        excel_file.save(excel_path)
        
        logger.info(f"Processing: PDF={pdf_filename}, Template={excel_filename}")
        
        # Read Excel headers (schema)
        fields = read_excel_headers(excel_path)
        logger.info(f"Fields to extract: {fields}")
        
        # Extract text from PDF
        pdf_text = extract_text_from_pdf(pdf_path)
        pdf_text = clean_extracted_text(pdf_text)
        
        # Build messages and run extraction
        if use_vision:
            images = pdf_to_images(pdf_path)
            messages = build_vision_messages(fields, images, custom_prompt, pdf_text)
            response = extract_with_vision(messages)
        else:
            messages = build_text_only_messages(fields, pdf_text, custom_prompt)
            response = extract_with_text_only(messages)
        
        logger.info(f"Model response length: {len(response)}")
        
        # Parse response to structured data
        extracted_rows = parse_model_response_to_rows(response, fields)
        
        # Create output Excel
        output_filename = generate_output_filename(pdf_file.filename)
        output_path = Path(app.config['OUTPUT_FOLDER']) / output_filename
        create_output_excel(excel_path, output_path, extracted_rows)
        
        # Cleanup uploaded files
        pdf_path.unlink(missing_ok=True)
        excel_path.unlink(missing_ok=True)
        
        if return_json:
            return jsonify({
                "success": True,
                "fields": fields,
                "extracted_data": extracted_rows,
                "download_url": f"/api/download/{output_filename}"
            })
        
        # Return the Excel file
        return send_file(
            output_path,
            as_attachment=True,
            download_name=output_filename,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
        
    except Exception as e:
        logger.exception("Extraction failed")
        return jsonify({"error": str(e)}), 500


@app.route('/api/download/<filename>', methods=['GET'])
def download_file(filename):
    """Download a previously generated output file."""
    file_path = Path(app.config['OUTPUT_FOLDER']) / secure_filename(filename)
    if not file_path.exists():
        return jsonify({"error": "File not found"}), 404
    return send_file(file_path, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

