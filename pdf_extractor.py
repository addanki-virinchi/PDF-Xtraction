"""PDF text extraction module with CPU optimization support."""

import logging
from pathlib import Path
from typing import List, Optional, Union

import pdfplumber
from pdf2image import convert_from_path, convert_from_bytes
from PIL import Image

from config import PDF_DPI, PDF_MAX_DIMENSION, PDF_MAX_PAGES

logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_path: Union[str, Path]) -> str:
    """
    Extract text content from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text content as a string
    """
    pdf_path = Path(pdf_path)
    text_content = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            page_text = page.extract_text()
            if page_text:
                text_content.append(f"--- Page {page_num} ---\n{page_text}")
    
    return "\n\n".join(text_content)


def extract_tables_from_pdf(pdf_path: Union[str, Path]) -> List[List[List[str]]]:
    """
    Extract tables from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        List of tables, where each table is a list of rows
    """
    pdf_path = Path(pdf_path)
    all_tables = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            if tables:
                all_tables.extend(tables)
    
    return all_tables


def resize_image_for_cpu(
    image: Image.Image,
    max_dimension: int = 800
) -> Image.Image:
    """
    Resize image to reduce memory usage and speed up CPU inference.

    Args:
        image: PIL Image to resize
        max_dimension: Maximum dimension (width or height)

    Returns:
        Resized PIL Image (or original if already smaller)
    """
    width, height = image.size

    # Skip if already within limits
    if width <= max_dimension and height <= max_dimension:
        return image

    # Calculate scale factor to fit within max_dimension
    scale = min(max_dimension / width, max_dimension / height)
    new_width = int(width * scale)
    new_height = int(height * scale)

    # Use LANCZOS for high-quality downsampling
    resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    logger.debug(f"Resized image from {width}x{height} to {new_width}x{new_height}")

    return resized


def pdf_to_images(
    pdf_path: Union[str, Path, bytes],
    dpi: Optional[int] = None,
    max_dimension: Optional[int] = None,
    max_pages: Optional[int] = None
) -> List[Image.Image]:
    """
    Convert PDF pages to PIL Images for vision model processing.

    Uses CPU-optimized settings from config.py by default.

    Args:
        pdf_path: Path to the PDF file or bytes content
        dpi: Resolution for image conversion (default: from config)
        max_dimension: Max image dimension for resizing (default: from config)
        max_pages: Maximum pages to process (default: from config, 0=all)

    Returns:
        List of PIL Image objects, one per page (resized for CPU efficiency)
    """
    # Use config defaults if not specified
    effective_dpi = dpi if dpi is not None else PDF_DPI
    effective_max_dim = max_dimension if max_dimension is not None else PDF_MAX_DIMENSION
    effective_max_pages = max_pages if max_pages is not None else PDF_MAX_PAGES

    logger.info(f"PDF to images: DPI={effective_dpi}, max_dim={effective_max_dim}, max_pages={effective_max_pages or 'all'}")

    # Convert PDF to images
    if isinstance(pdf_path, bytes):
        images = convert_from_bytes(pdf_path, dpi=effective_dpi)
    else:
        images = convert_from_path(str(pdf_path), dpi=effective_dpi)

    total_pages = len(images)
    logger.info(f"PDF has {total_pages} pages")

    # Limit pages if configured
    if effective_max_pages > 0 and len(images) > effective_max_pages:
        logger.warning(f"Limiting to first {effective_max_pages} of {total_pages} pages for faster processing")
        images = images[:effective_max_pages]

    # Resize images for CPU efficiency
    if effective_max_dim > 0:
        resized_images = []
        for i, img in enumerate(images):
            original_size = img.size
            resized = resize_image_for_cpu(img, effective_max_dim)
            if resized.size != original_size:
                logger.debug(f"Page {i+1}: {original_size} -> {resized.size}")
            resized_images.append(resized)
        images = resized_images

    return images


def get_pdf_page_count(pdf_path: Union[str, Path]) -> int:
    """Get the number of pages in a PDF file."""
    with pdfplumber.open(pdf_path) as pdf:
        return len(pdf.pages)


def clean_extracted_text(text: str) -> str:
    """
    Clean and normalize extracted PDF text.
    
    Args:
        text: Raw extracted text
        
    Returns:
        Cleaned text
    """
    # Remove excessive whitespace while preserving structure
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        # Strip trailing/leading whitespace from each line
        cleaned_line = line.strip()
        # Keep non-empty lines or preserve paragraph breaks
        if cleaned_line or (cleaned_lines and cleaned_lines[-1]):
            cleaned_lines.append(cleaned_line)
    
    # Join and remove excessive blank lines
    result = '\n'.join(cleaned_lines)
    
    # Replace multiple consecutive newlines with double newline
    while '\n\n\n' in result:
        result = result.replace('\n\n\n', '\n\n')
    
    return result.strip()

