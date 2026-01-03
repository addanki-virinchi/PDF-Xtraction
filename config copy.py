"""Configuration settings for the PDF Extraction application."""

import ctypes
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional

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
#   - "Qwen/Qwen2-VL-2B-Instruct"   (~4.5GB download, ~5-6GB RAM) - Best for 16GB RAM
#   - "Qwen/Qwen2-VL-7B-Instruct"   (~15GB download, ~14-16GB RAM) - Needs 24GB+ RAM
#   - "Qwen/Qwen2.5-VL-7B-Instruct" (~15GB download, ~14-16GB RAM) - Needs 24GB+ RAM
#   - "Qwen/Qwen3-VL-32B-Instruct"  (~66GB download, ~17GB RAM with 4bit) - Needs GPU
DEFAULT_MODEL = "Qwen/Qwen2-VL-7B-Instruct"  # Best for 16GB RAM CPU-only systems
MODEL_NAME = os.getenv("MODEL_NAME", DEFAULT_MODEL)

logger = logging.getLogger(__name__)

KNOWN_QWEN_MODELS = {
    "Qwen/Qwen2-VL-2B-Instruct",
    "Qwen/Qwen2-VL-7B-Instruct",
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "Qwen/Qwen3-VL-32B-Instruct",
}

MODEL_PROFILES: Dict[str, Dict[str, Any]] = {
    "2b": {
        "label": "2B",
        "size_b": 2,
        "min_ram_gb": 8,
        "recommended_ram_gb": 12,
        "requires_gpu": False,
        "max_new_tokens": 4096,
        "pdf_dpi": 120,
        "pdf_max_dimension": 1024,
        "pdf_max_pages": 0,
        "quantization_mode": "none",
        "low_cpu_mem_usage": True,
        "generation": {
            "do_sample": True,
            "top_p": 0.8,
            "top_k": 20,
            "temperature": 0.7,
            "repetition_penalty": 1.0,
        },
    },
    "7b": {
        "label": "7B",
        "size_b": 7,
        "min_ram_gb": 14,
        "recommended_ram_gb": 24,
        "requires_gpu": False,
        "max_new_tokens": 3072,
        "pdf_dpi": 110,
        "pdf_max_dimension": 960,
        "pdf_max_pages": 0,
        "quantization_mode": "none",
        "low_cpu_mem_usage": True,
        "generation": {
            "do_sample": True,
            "top_p": 0.8,
            "top_k": 20,
            "temperature": 0.7,
            "repetition_penalty": 1.0,
        },
    },
    "32b": {
        "label": "32B",
        "size_b": 32,
        "min_ram_gb": 24,
        "recommended_ram_gb": 48,
        "requires_gpu": True,
        "max_new_tokens": 2048,
        "pdf_dpi": 90,
        "pdf_max_dimension": 768,
        "pdf_max_pages": 0,
        "quantization_mode": "4bit",
        "low_cpu_mem_usage": True,
        "generation": {
            "do_sample": True,
            "top_p": 0.8,
            "top_k": 20,
            "temperature": 0.6,
            "repetition_penalty": 1.05,
        },
    },
    "unknown": {
        "label": "Unknown",
        "size_b": None,
        "min_ram_gb": 0,
        "recommended_ram_gb": 0,
        "requires_gpu": False,
        "max_new_tokens": 3072,
        "pdf_dpi": 110,
        "pdf_max_dimension": 960,
        "pdf_max_pages": 0,
        "quantization_mode": "none",
        "low_cpu_mem_usage": True,
        "generation": {
            "do_sample": True,
            "top_p": 0.8,
            "top_k": 20,
            "temperature": 0.7,
            "repetition_penalty": 1.0,
        },
    },
}


def _get_system_memory_gb() -> float:
    """Best-effort total system RAM (GB)."""
    try:
        import psutil  # type: ignore
        return round(psutil.virtual_memory().total / 1024**3, 1)
    except Exception:
        pass

    if hasattr(os, "sysconf"):
        try:
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return round((pages * page_size) / 1024**3, 1)
        except Exception:
            return 0.0

    if sys.platform.startswith("win"):
        try:
            class _MEMORYSTATUS(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = _MEMORYSTATUS()
            status.dwLength = ctypes.sizeof(_MEMORYSTATUS)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return round(status.ullTotalPhys / 1024**3, 1)
        except Exception:
            return 0.0

    return 0.0


def _detect_model_size_b(model_name: str) -> Optional[float]:
    match = re.search(r"-(\d+(?:\.\d+)?)B-", model_name, re.IGNORECASE)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _resolve_model_profile(model_name: str) -> Dict[str, Any]:
    size_b = _detect_model_size_b(model_name)
    if size_b is None:
        return MODEL_PROFILES["unknown"]
    if size_b <= 3:
        return MODEL_PROFILES["2b"]
    if size_b <= 8:
        return MODEL_PROFILES["7b"]
    if size_b >= 24:
        return MODEL_PROFILES["32b"]
    return MODEL_PROFILES["unknown"]


def _get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning(f"Invalid int for {name}={value!r}. Using default {default}.")
        return default


def _get_env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.lower() == "true"


SYSTEM_RAM_GB = _get_system_memory_gb()
MODEL_PROFILE = _resolve_model_profile(MODEL_NAME)
MODEL_SIZE_B = MODEL_PROFILE.get("size_b")
MODEL_IS_KNOWN = MODEL_NAME in KNOWN_QWEN_MODELS

MAX_NEW_TOKENS = _get_env_int("MAX_NEW_TOKENS", MODEL_PROFILE["max_new_tokens"])
USE_FLASH_ATTENTION_ENV = os.getenv("USE_FLASH_ATTENTION")
USE_FLASH_ATTENTION = _get_env_bool("USE_FLASH_ATTENTION", False)

# =============================================================================
# CPU Optimization Settings
# =============================================================================
# These settings significantly impact processing speed on CPU

# PDF Image Processing - Lower values = faster but potentially less accurate
# DPI: 72=fast/low quality, 100=balanced, 150=high quality/slow
PDF_DPI = _get_env_int("PDF_DPI", MODEL_PROFILE["pdf_dpi"])

# Max dimension: Resize images so longest side doesn't exceed this
# 640=very fast, 800=fast, 1024=balanced, 2048=high quality/slow
PDF_MAX_DIMENSION = _get_env_int("PDF_MAX_DIMENSION", MODEL_PROFILE["pdf_max_dimension"])

# Max pages: Limit number of pages to process (0 = no limit)
# For long PDFs, processing only first N pages can dramatically speed up extraction
PDF_MAX_PAGES = _get_env_int("PDF_MAX_PAGES", MODEL_PROFILE["pdf_max_pages"])

# Quantization settings to reduce memory usage
# Options: "none", "4bit", "8bit"
# - "none": Full precision (recommended for 2B model on CPU)
# - "8bit": ~50% memory reduction (requires bitsandbytes + GPU)
# - "4bit": ~75% memory reduction (requires bitsandbytes + GPU)
# Note: Quantization requires GPU - use "none" for CPU-only
QUANTIZATION_MODE_ENV = os.getenv("QUANTIZATION_MODE")
QUANTIZATION_MODE = (QUANTIZATION_MODE_ENV or MODEL_PROFILE["quantization_mode"]).lower()

# Memory optimization settings
LOW_CPU_MEM_USAGE = _get_env_bool("LOW_CPU_MEM_USAGE", MODEL_PROFILE["low_cpu_mem_usage"])

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
    **MODEL_PROFILE["generation"],
}

# Flask configuration
class FlaskConfig:
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB max file size
    UPLOAD_FOLDER = str(UPLOAD_FOLDER)
    OUTPUT_FOLDER = str(OUTPUT_FOLDER)

