#!/usr/bin/env python
"""
Pre-download Qwen2-VL model and processor to local cache.

This script downloads the model weights and processor/tokenizer files
to the configured HF_CACHE_DIR before running the Flask application,
so that the first run of app.py doesn't have any download delay.

Usage:
    python download_model.py                    # Download default model
    python download_model.py --model Qwen/Qwen2-VL-7B-Instruct  # Download specific model
    python download_model.py --verify-only      # Only verify existing download
"""

import argparse
import sys
import os
from pathlib import Path

# Import config first to set up environment variables (cache directories, etc.)
# This MUST happen before importing transformers
import config

print("=" * 60)
print("Qwen VL Model Pre-Download Script")
print("=" * 60)
print(f"\nConfiguration:")
print(f"  Cache directory: {config.HF_CACHE_DIR}")
print(f"  Default model:   {config.MODEL_NAME}")
print(f"  Temp directory:  {config.TEMP_DIR}")
print()


def get_model_class(model_name: str):
    """Get the appropriate model class based on model name."""
    model_lower = model_name.lower()
    
    if "qwen3" in model_lower:
        from transformers import Qwen3VLForConditionalGeneration
        return Qwen3VLForConditionalGeneration
    elif "qwen2.5" in model_lower or "qwen2_5" in model_lower:
        from transformers import Qwen2_5_VLForConditionalGeneration
        return Qwen2_5_VLForConditionalGeneration
    else:
        from transformers import Qwen2VLForConditionalGeneration
        return Qwen2VLForConditionalGeneration


def download_model(model_name: str, verify_only: bool = False):
    """Download or verify the model and processor."""
    
    print(f"Model: {model_name}")
    print("-" * 60)
    
    # Import transformers components
    from transformers import AutoProcessor
    from huggingface_hub import snapshot_download, HfApi
    import torch
    
    try:
        # Step 1: Check if model exists on Hugging Face
        print("\n[1/4] Checking model availability on Hugging Face...")
        api = HfApi()
        try:
            model_info = api.model_info(model_name)
            print(f"  ✓ Model found: {model_info.id}")
            print(f"  ✓ Downloads: {model_info.downloads:,}")
        except Exception as e:
            print(f"  ✗ Error: Could not find model '{model_name}' on Hugging Face")
            print(f"    {e}")
            return False
        
        # Step 2: Download/verify processor
        print("\n[2/4] Downloading processor/tokenizer...")
        try:
            processor = AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            print(f"  ✓ Processor loaded successfully")
        except Exception as e:
            print(f"  ✗ Error downloading processor: {e}")
            return False
        
        if verify_only:
            print("\n[3/4] Skipping model download (verify-only mode)")
            print("\n[4/4] Verifying cached model files...")
        else:
            # Step 3: Download model weights
            print("\n[3/4] Downloading model weights (this may take a while)...")
            print("      Progress will be shown below:")
            print()
            
            try:
                # Use snapshot_download for better progress display
                local_path = snapshot_download(
                    repo_id=model_name,
                    cache_dir=config.HF_CACHE_DIR,
                    resume_download=True,
                    local_files_only=False
                )
                print(f"\n  ✓ Model downloaded to: {local_path}")
            except Exception as e:
                print(f"\n  ✗ Error downloading model: {e}")
                return False
            
            # Step 4: Verify by loading the model
            print("\n[4/4] Verifying model can be loaded...")
        
        try:
            model_class = get_model_class(model_name)
            print(f"  Using model class: {model_class.__name__}")
            
            # Quick verification - just check files exist, don't fully load on CPU
            # Full loading would take too long and use too much memory
            from huggingface_hub import hf_hub_download
            config_file = hf_hub_download(
                repo_id=model_name,
                filename="config.json",
                cache_dir=config.HF_CACHE_DIR,
                local_files_only=True  # Should be cached now
            )
            print(f"  ✓ Model config verified at: {config_file}")
            
        except Exception as e:
            print(f"  ✗ Verification failed: {e}")
            return False
        
        print("\n" + "=" * 60)
        print("✓ SUCCESS: Model and processor are ready!")
        print("=" * 60)
        print(f"\nYou can now run 'python app.py' without download delays.")
        print(f"The model will be loaded from: {config.HF_CACHE_DIR}")
        return True
        
    except KeyboardInterrupt:
        print("\n\n✗ Download cancelled by user.")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Pre-download Qwen VL model to local cache"
    )
    parser.add_argument(
        "--model", "-m",
        default=config.MODEL_NAME,
        help=f"Model name to download (default: {config.MODEL_NAME})"
    )
    parser.add_argument(
        "--verify-only", "-v",
        action="store_true",
        help="Only verify existing download, don't download new files"
    )
    
    args = parser.parse_args()
    
    success = download_model(args.model, args.verify_only)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

