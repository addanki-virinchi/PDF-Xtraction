"""Configuration settings for the PDF Extraction application."""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent

# Upload and output directories
UPLOAD_FOLDER = BASE_DIR / "uploads"
OUTPUT_FOLDER = BASE_DIR / "outputs"

# Ensure directories exist
UPLOAD_FOLDER.mkdir(exist_ok=True)
OUTPUT_FOLDER.mkdir(exist_ok=True)

# Temp directory for model loading and large file operations
# Set to D: drive to avoid filling C: drive SSD
# This affects Hugging Face cache and PyTorch temp files
#if windows
# TEMP_DIR = os.getenv("TEMP_DIR", "D:\\temp\\pdf_extraction")
# HF_CACHE_DIR = os.getenv("HF_HOME", "D:\\huggingface_cache")
# Temp directory for model loading and large file operations
TEMP_DIR = os.getenv("TEMP_DIR", "/workspace/tmp/pdf_extraction")
HF_CACHE_DIR = os.getenv("HF_HOME", "/workspace/hf_cache")

# Create temp directories if they don't exist
Path(TEMP_DIR).mkdir(parents=True, exist_ok=True)
Path(HF_CACHE_DIR).mkdir(parents=True, exist_ok=True)

# Set environment variables for Hugging Face and PyTorch
# These must be set before importing transformers/torch
os.environ["TMPDIR"] = TEMP_DIR
os.environ["TEMP"] = TEMP_DIR
os.environ["TMP"] = TEMP_DIR
os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = os.path.join(HF_CACHE_DIR, "transformers")
os.environ["HF_DATASETS_CACHE"] = os.path.join(HF_CACHE_DIR, "datasets")

# Enable faster downloads with hf_transfer (if installed)
# Install with: pip install hf_transfer
# Can provide 3-5x faster download speeds
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = os.getenv("HF_HUB_ENABLE_HF_TRANSFER", "1")

# Allowed file extensions
ALLOWED_PDF_EXTENSIONS = {'.pdf'}
ALLOWED_EXCEL_EXTENSIONS = {'.xlsx', '.xls'}

# Model configuration
# Available Qwen VL models (set via MODEL_NAME environment variable):
#   - "Qwen/Qwen2-VL-2B-Instruct"    (~4.5GB download, ~5-6GB RAM) - Best for 16GB RAM
#   - "Qwen/Qwen2.5-VL-3B-Instruct"  (~7.5GB download, ~8-10GB RAM) - Faster than 7B
#   - "Qwen/Qwen3-VL-4B-Instruct"    (~9-10GB download, ~10-12GB RAM) - Balanced speed/quality
#   - "Qwen/Qwen2-VL-7B-Instruct"    (~15GB download, ~14-16GB RAM) - Needs 24GB+ RAM
#   - "Qwen/Qwen2.5-VL-7B-Instruct"  (~15GB download, ~14-16GB RAM) - Needs 24GB+ RAM
#   - "Qwen/Qwen3-VL-32B-Instruct"   (~66GB download, ~17GB RAM with 4bit) - Needs GPU
DEFAULT_MODEL = "Qwen/Qwen3-VL-4B-Instruct"  # Balanced default for multi-page PDFs
MODEL_NAME = os.getenv("MODEL_NAME", DEFAULT_MODEL)

# Generation settings
# MAX_NEW_TOKENS controls how much text the model generates
# Lower values = faster processing, higher values = more complete responses
# For CPU: 2048-4096 is recommended (faster), for GPU: 8192+ is fine
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "3072"))  # Lower default for faster 3B inference
USE_FLASH_ATTENTION = os.getenv("USE_FLASH_ATTENTION", "false").lower() == "true"

# =============================================================================
# CPU Optimization Settings
# =============================================================================
# These settings significantly impact processing speed on CPU

# PDF Image Processing - Lower values = faster but potentially less accurate
# DPI: 72=fast/low quality, 100=balanced, 150=high quality/slow
PDF_DPI = int(os.getenv("PDF_DPI", "100"))  # Default 100 for CPU (was 150)

# Max dimension: Resize images so longest side doesn't exceed this
# 640=very fast, 800=fast, 1024=balanced, 2048=high quality/slow
PDF_MAX_DIMENSION = int(os.getenv("PDF_MAX_DIMENSION", "1024"))  # Default 800 for CPU

# Max pages: Limit number of pages to process (0 = no limit)
# For long PDFs, processing only first N pages can dramatically speed up extraction
PDF_MAX_PAGES = int(os.getenv("PDF_MAX_PAGES", "0"))  # Default: no limit

# Quantization settings to reduce memory usage
# Options: "none", "4bit", "8bit"
# - "none": Full precision (recommended for 2B model on CPU)
# - "8bit": ~50% memory reduction (requires bitsandbytes + GPU)
# - "4bit": ~75% memory reduction (requires bitsandbytes + GPU)
# Note: Quantization requires GPU - use "none" for CPU-only
QUANTIZATION_MODE = os.getenv("QUANTIZATION_MODE", "none").lower()

# Memory optimization settings
LOW_CPU_MEM_USAGE = os.getenv("LOW_CPU_MEM_USAGE", "true").lower() == "true"

# CPU-specific settings
# Number of threads for PyTorch CPU operations
# Using too many threads can cause system hangs
# Recommended: 4 threads for 16GB RAM systems (balances speed vs responsiveness)
CPU_THREADS = int(os.getenv("CPU_THREADS", "5"))  # Default: 4 (not auto-detect)
if CPU_THREADS == 0:
    import multiprocessing
    # If explicitly set to 0, use half of available cores
    torch_threads = max(1, multiprocessing.cpu_count() // 2)
else:
    torch_threads = CPU_THREADS

os.environ["OMP_NUM_THREADS"] = str(torch_threads)
os.environ["MKL_NUM_THREADS"] = str(torch_threads)

# Additional CPU optimization: enable memory-efficient settings
# These help prevent system hangs during inference
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # Fallback for Apple Silicon
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid tokenizer threading issues

# Generation parameters for Vision-Language model
VL_GENERATION_CONFIG = {
    "max_new_tokens": MAX_NEW_TOKENS,
    # Deterministic decoding for speed + stability on structured extraction.
    "do_sample": False,
    "top_p": 1.0,
    "top_k": 0,
    "temperature": 0.0,
    "repetition_penalty": 1.0,
}

# Flask configuration
class FlaskConfig:
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB max file size
    UPLOAD_FOLDER = str(UPLOAD_FOLDER)
    OUTPUT_FOLDER = str(OUTPUT_FOLDER)

