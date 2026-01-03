"""QWEN Vision-Language Model integration module."""

import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image

from config import (
    MODEL_NAME,
    MODEL_PROFILE,
    MODEL_SIZE_B,
    MODEL_IS_KNOWN,
    SYSTEM_RAM_GB,
    VL_GENERATION_CONFIG,
    USE_FLASH_ATTENTION,
    USE_FLASH_ATTENTION_ENV,
    QUANTIZATION_MODE,
    QUANTIZATION_MODE_ENV,
    LOW_CPU_MEM_USAGE
)

logger = logging.getLogger(__name__)

# Global model instance (singleton pattern for efficiency)
_model = None
_processor = None
_loading_lock = threading.Lock()
_loading_status = {
    "state": "not_started",  # not_started, loading, ready, error
    "progress": "",
    "started_at": None,
    "completed_at": None,
    "error": None,
    "warnings": [],
    "suggestions": []
}
_effective_quantization_mode = QUANTIZATION_MODE


def get_loading_status() -> Dict[str, Any]:
    """Get the current model loading status."""
    status = _loading_status.copy()
    if status["started_at"] and not status["completed_at"]:
        status["elapsed_seconds"] = time.time() - status["started_at"]
    return status


def is_model_ready() -> bool:
    """Check if the model is loaded and ready for inference."""
    return _loading_status["state"] == "ready" and _model is not None


def _model_cache_dir(model_name: str) -> Optional[Path]:
    cache_root = os.environ.get("TRANSFORMERS_CACHE") or os.environ.get("HF_HOME")
    if not cache_root:
        return None
    org_repo = model_name.replace("/", "--")
    return Path(cache_root) / f"models--{org_repo}"


def _is_model_cached(model_name: str) -> bool:
    cache_dir = _model_cache_dir(model_name)
    if not cache_dir:
        return False
    return cache_dir.exists()


def _suggest_alternatives(has_gpu: bool) -> List[str]:
    if has_gpu:
        return [
            "Qwen/Qwen2-VL-7B-Instruct",
            "Qwen/Qwen2-VL-2B-Instruct",
        ]
    return [
        "Qwen/Qwen2-VL-2B-Instruct",
        "Qwen/Qwen2-VL-7B-Instruct",
    ]


def _validate_model_configuration(has_gpu: bool) -> List[str]:
    warnings: List[str] = []

    if not MODEL_IS_KNOWN:
        warnings.append(
            f"Model {MODEL_NAME} is not in the known Qwen list; defaults may be suboptimal."
        )

    if MODEL_PROFILE.get("requires_gpu") and not has_gpu:
        raise RuntimeError(
            f"Model {MODEL_NAME} requires a GPU, but none is available. "
            f"Suggested models: {', '.join(_suggest_alternatives(has_gpu=False))}."
        )

    if SYSTEM_RAM_GB and MODEL_PROFILE.get("min_ram_gb"):
        min_ram = MODEL_PROFILE["min_ram_gb"]
        rec_ram = MODEL_PROFILE.get("recommended_ram_gb", min_ram)
        if SYSTEM_RAM_GB < min_ram and not has_gpu:
            raise RuntimeError(
                f"System RAM {SYSTEM_RAM_GB}GB is below the minimum {min_ram}GB for {MODEL_NAME}. "
                f"Suggested models: {', '.join(_suggest_alternatives(has_gpu=False))}."
            )
        if SYSTEM_RAM_GB < rec_ram and not has_gpu:
            warnings.append(
                f"System RAM {SYSTEM_RAM_GB}GB is below recommended {rec_ram}GB for {MODEL_NAME}; "
                "performance may be slow."
            )

    offline = os.getenv("HF_HUB_OFFLINE", "").lower() in ("1", "true") or \
        os.getenv("TRANSFORMERS_OFFLINE", "").lower() in ("1", "true")
    if offline and not _is_model_cached(MODEL_NAME):
        raise RuntimeError(
            f"Offline mode is enabled and {MODEL_NAME} is not cached locally. "
            f"Disable offline mode or pre-download the model. "
            f"Suggested models: {', '.join(_suggest_alternatives(has_gpu))}."
        )

    return warnings


def load_model(force_reload: bool = False) -> Tuple[Any, Any]:
    """
    Load the QWEN VL model and processor.

    Uses singleton pattern to avoid reloading the model on each request.

    Args:
        force_reload: If True, reload the model even if already loaded

    Returns:
        Tuple of (model, processor)

    Raises:
        RuntimeError: If model loading fails
    """
    global _model, _processor, _loading_status

    with _loading_lock:
        if _model is not None and _processor is not None and not force_reload:
            return _model, _processor

        if _loading_status["state"] == "loading":
            logger.warning("Model is already being loaded by another thread")
            # Wait for loading to complete
            while _loading_status["state"] == "loading":
                time.sleep(1)
            if _loading_status["state"] == "ready":
                return _model, _processor
            else:
                raise RuntimeError(f"Model loading failed: {_loading_status.get('error')}")

        _loading_status = {
            "state": "loading",
            "progress": "Initializing...",
            "started_at": time.time(),
            "completed_at": None,
            "error": None,
            "warnings": [],
            "suggestions": []
        }

    try:
        logger.info(f"Loading model: {MODEL_NAME}")
        logger.info(f"Quantization mode: {QUANTIZATION_MODE}")
        logger.info(f"Low CPU memory usage: {LOW_CPU_MEM_USAGE}")
        _loading_status["progress"] = "Importing transformers..."

        from transformers import AutoProcessor

        # Try to import BitsAndBytesConfig for quantization (optional)
        BitsAndBytesConfig = None
        if QUANTIZATION_MODE in ("4bit", "8bit"):
            try:
                from transformers import BitsAndBytesConfig as BnBConfig
                BitsAndBytesConfig = BnBConfig
                logger.info("BitsAndBytesConfig available for quantization")
            except ImportError:
                logger.warning(
                    "BitsAndBytesConfig not available. Install bitsandbytes for quantization. "
                    "Falling back to full precision."
                )

        # Determine model class based on model name
        # Qwen2-VL and Qwen2.5-VL use Qwen2VLForConditionalGeneration
        # Qwen3-VL uses Qwen3VLForConditionalGeneration (requires git transformers)
        is_qwen3 = "qwen3" in MODEL_NAME.lower()

        if is_qwen3:
            try:
                from transformers import Qwen3VLForConditionalGeneration
                model_class = Qwen3VLForConditionalGeneration
                logger.info("Using Qwen3VLForConditionalGeneration")
            except ImportError:
                raise ImportError(
                    "Qwen3VLForConditionalGeneration not available. "
                    "Qwen3-VL requires transformers from git main. "
                    "Use Qwen2-VL models instead (set MODEL_NAME=Qwen/Qwen2-VL-7B-Instruct)"
                )
        else:
            from transformers import Qwen2VLForConditionalGeneration
            model_class = Qwen2VLForConditionalGeneration
            logger.info("Using Qwen2VLForConditionalGeneration")

        # Detect if GPU is available
        has_gpu = torch.cuda.is_available()
        logger.info(f"GPU available: {has_gpu}")

        if has_gpu:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
        else:
            logger.info("Running on CPU - model will be slower but functional")
            # Set CPU-specific optimizations
            torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 4)))
            logger.info(f"CPU threads: {torch.get_num_threads()}")

        # Validate model/system compatibility
        warnings = _validate_model_configuration(has_gpu)
        if warnings:
            _loading_status["warnings"] = warnings
            for warning in warnings:
                logger.warning(warning)

        # Build loading configuration
        load_kwargs = {
            "low_cpu_mem_usage": LOW_CPU_MEM_USAGE,
        }

        # Device mapping: auto for GPU, cpu for CPU-only
        if has_gpu:
            load_kwargs["device_map"] = "auto"
        else:
            # For CPU-only, don't use device_map - load directly to CPU
            load_kwargs["device_map"] = "cpu"
            # Use float32 for CPU (bfloat16 may not be supported on all CPUs)
            load_kwargs["torch_dtype"] = torch.float32
            logger.info("Using CPU with float32 precision")

        # Configure quantization (requires GPU - skip for CPU)
        global _effective_quantization_mode
        _effective_quantization_mode = QUANTIZATION_MODE
        if not has_gpu and QUANTIZATION_MODE in ("4bit", "8bit"):
            _effective_quantization_mode = "none"
            logger.warning(
                f"Quantization {QUANTIZATION_MODE} requires GPU; falling back to full precision."
            )

        if has_gpu and _effective_quantization_mode == "4bit" and BitsAndBytesConfig:
            _loading_status["progress"] = "Configuring 4-bit quantization..."
            logger.info("Using 4-bit quantization")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            load_kwargs["quantization_config"] = quantization_config
        elif has_gpu and _effective_quantization_mode == "8bit" and BitsAndBytesConfig:
            _loading_status["progress"] = "Configuring 8-bit quantization..."
            logger.info("Using 8-bit quantization")
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
            load_kwargs["quantization_config"] = quantization_config
        elif has_gpu:
            # GPU with full precision
            logger.info("Using GPU with full precision")
            load_kwargs["torch_dtype"] = torch.bfloat16 if USE_FLASH_ATTENTION else "auto"

        # Add flash attention if enabled (GPU only)
        if USE_FLASH_ATTENTION and has_gpu:
            logger.info("Using Flash Attention 2")
            load_kwargs["attn_implementation"] = "flash_attention_2"
            if "torch_dtype" not in load_kwargs:
                load_kwargs["torch_dtype"] = torch.bfloat16
        elif not has_gpu:
            logger.info("Flash Attention not available on CPU")

        quant_info = (
            f"({_effective_quantization_mode})"
            if _effective_quantization_mode != "none"
            else "(full precision)"
        )
        _loading_status["progress"] = f"Downloading/loading model weights {quant_info}..."
        logger.info("Starting model download/loading - this may take several minutes")

        load_start = time.time()

        try:
            _model = model_class.from_pretrained(MODEL_NAME, **load_kwargs)
        except OSError as e:
            if "paging file" in str(e).lower() or "1455" in str(e):
                error_msg = (
                    f"Insufficient virtual memory (Windows error 1455). "
                    f"Model: {MODEL_NAME}, Quantization: {_effective_quantization_mode}. "
                    f"Solutions: 1) Use smaller model (MODEL_NAME=Qwen/Qwen2-VL-2B-Instruct), "
                    f"2) Set QUANTIZATION_MODE=4bit (GPU only), "
                    f"3) Increase Windows page file. See logs for details."
                )
                logger.error(error_msg)
                raise OSError(error_msg) from e
            raise

        load_duration = time.time() - load_start
        logger.info(f"Model weights loaded in {load_duration:.1f} seconds")

        _loading_status["progress"] = "Loading processor..."
        _processor = AutoProcessor.from_pretrained(MODEL_NAME)

        with _loading_lock:
            _loading_status["state"] = "ready"
            _loading_status["progress"] = "Model ready"
            _loading_status["completed_at"] = time.time()

        total_duration = _loading_status["completed_at"] - _loading_status["started_at"]
        logger.info(f"Model loaded successfully in {total_duration:.1f} seconds")

        # Log memory usage
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                logger.info(f"GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

        return _model, _processor

    except Exception as e:
        with _loading_lock:
            _loading_status["state"] = "error"
            _loading_status["error"] = str(e)
            _loading_status["suggestions"] = _suggest_alternatives(torch.cuda.is_available())
            _loading_status["completed_at"] = time.time()
        logger.exception("Failed to load model")
        raise


def preload_model_async():
    """Start loading the model in a background thread."""
    if _loading_status["state"] in ("loading", "ready"):
        logger.info(f"Model already {_loading_status['state']}, skipping preload")
        return

    def _load():
        try:
            load_model()
        except Exception as e:
            logger.error(f"Background model loading failed: {e}")

    thread = threading.Thread(target=_load, name="ModelPreloader", daemon=True)
    thread.start()
    logger.info("Started background model preloading thread")


def extract_with_vision(
    messages: List[Dict],
    generation_config: Optional[Dict] = None
) -> str:
    """
    Run extraction using the vision-language model.

    Args:
        messages: List of message dictionaries with text and image content
        generation_config: Optional override for generation parameters

    Returns:
        Model response text
    """
    import gc

    model, processor = load_model()

    # Merge generation config
    config = VL_GENERATION_CONFIG.copy()
    if generation_config:
        config.update(generation_config)

    # Log inference start for timing
    inference_start = time.time()
    logger.info(f"Starting inference with max_new_tokens={config.get('max_new_tokens', 'default')}")

    # Count images in messages for logging
    image_count = 0
    for msg in messages:
        if isinstance(msg.get("content"), list):
            image_count += sum(1 for c in msg["content"] if c.get("type") == "image")
    logger.info(f"Processing {image_count} images")

    # Prepare inputs
    logger.debug("Preparing inputs...")
    prep_start = time.time()
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)
    logger.debug(f"Input preparation took {time.time() - prep_start:.1f}s")

    # Log input size
    input_tokens = inputs.input_ids.shape[1] if hasattr(inputs, 'input_ids') else 0
    logger.info(f"Input tokens: {input_tokens}")

    # Generate response with memory optimization
    # Use inference_mode for better CPU performance (slightly faster than no_grad)
    logger.info("Starting model generation (this may take several minutes on CPU)...")
    gen_start = time.time()

    try:
        with torch.inference_mode():
            generated_ids = model.generate(**inputs, **config)
    finally:
        # Clear input tensors to free memory during decoding
        del inputs
        gc.collect()

    gen_duration = time.time() - gen_start
    output_tokens = generated_ids.shape[1] - input_tokens if input_tokens > 0 else generated_ids.shape[1]
    tokens_per_sec = output_tokens / gen_duration if gen_duration > 0 else 0
    logger.info(f"Generation took {gen_duration:.1f}s ({output_tokens} tokens, {tokens_per_sec:.1f} tokens/sec)")

    # Trim input tokens from output
    generated_ids_trimmed = [
        out_ids[input_tokens:]
        for out_ids in generated_ids
    ]

    # Decode response
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    # Final cleanup
    del generated_ids, generated_ids_trimmed
    gc.collect()

    total_duration = time.time() - inference_start
    logger.info(f"Total inference time: {total_duration:.1f}s")

    return output_text[0] if output_text else ""


def extract_with_text_only(
    messages: List[Dict],
    generation_config: Optional[Dict] = None
) -> str:
    """
    Run extraction using text-only input (no images).
    
    Args:
        messages: List of message dictionaries with text content
        generation_config: Optional override for generation parameters
        
    Returns:
        Model response text
    """
    # For text-only, we can still use the same model
    return extract_with_vision(messages, generation_config)


def get_model_info() -> Dict[str, Any]:
    """Get information about the loaded model."""
    loading_status = get_loading_status()

    info = {
        "model_name": MODEL_NAME,
        "quantization_mode": QUANTIZATION_MODE,
        "effective_quantization_mode": _effective_quantization_mode,
        "loading_state": loading_status["state"],
        "loading_progress": loading_status["progress"],
        "model_profile": MODEL_PROFILE.get("label"),
        "model_size_b": MODEL_SIZE_B,
        "system_ram_gb": SYSTEM_RAM_GB,
        "model_known": MODEL_IS_KNOWN,
    }

    if loading_status["started_at"]:
        if loading_status["completed_at"]:
            info["load_time_seconds"] = loading_status["completed_at"] - loading_status["started_at"]
        else:
            info["elapsed_seconds"] = time.time() - loading_status["started_at"]

    if loading_status["error"]:
        info["error"] = loading_status["error"]

    if _model is not None:
        info["device"] = str(_model.device)
        info["dtype"] = str(_model.dtype)

        # Check if model is quantized
        if hasattr(_model, 'is_quantized'):
            info["is_quantized"] = _model.is_quantized

        # Add memory info if GPU available
        if torch.cuda.is_available():
            gpu_info = []
            for i in range(torch.cuda.device_count()):
                gpu_info.append({
                    "gpu": i,
                    "allocated_gb": round(torch.cuda.memory_allocated(i) / 1024**3, 2),
                    "reserved_gb": round(torch.cuda.memory_reserved(i) / 1024**3, 2)
                })
            info["gpu_memory"] = gpu_info

    return info

