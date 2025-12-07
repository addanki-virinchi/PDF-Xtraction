"""PDF text extraction module."""

import io
from pathlib import Path
from typing import List, Optional, Union

import pdfplumber
from pdf2image import convert_from_path, convert_from_bytes
from PIL import Image


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


def pdf_to_images(
    pdf_path: Union[str, Path, bytes],
    dpi: int = 150
) -> List[Image.Image]:
    """
    Convert PDF pages to PIL Images for vision model processing.
    
    Args:
        pdf_path: Path to the PDF file or bytes content
        dpi: Resolution for image conversion
        
    Returns:
        List of PIL Image objects, one per page
    """
    if isinstance(pdf_path, bytes):
        images = convert_from_bytes(pdf_path, dpi=dpi)
    else:
        images = convert_from_path(str(pdf_path), dpi=dpi)
    
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

