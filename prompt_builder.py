"""Prompt engineering module for PDF data extraction."""

from typing import List, Optional


DEFAULT_SYSTEM_PROMPT = """You are an expert data extraction assistant. Your task is to carefully analyze documents and extract specific information based on the requested fields.

Key Instructions:
1. Extract ONLY the information that matches the requested fields
2. Return data in valid JSON format
3. If a field value is not found in the document, use null
4. If multiple records/rows exist (e.g., line items in an invoice), return an array of objects
5. Preserve the exact field names as provided
6. Be precise and accurate - do not hallucinate or guess values
7. For dates, use ISO format (YYYY-MM-DD) when possible
8. For numbers, extract numeric values without currency symbols unless the field specifically asks for formatted values"""


def build_extraction_prompt(
    fields: List[str],
    pdf_text: Optional[str] = None,
    custom_prompt: Optional[str] = None,
    document_type: Optional[str] = None
) -> str:
    """
    Build a comprehensive extraction prompt for the model.
    
    Args:
        fields: List of field names to extract (from Excel headers)
        pdf_text: Optional text content from PDF (for text-based mode)
        custom_prompt: Optional additional instructions from user
        document_type: Optional hint about document type (invoice, contract, etc.)
        
    Returns:
        Formatted prompt string
    """
    # Build field list
    fields_formatted = "\n".join([f"- {field}" for field in fields])
    
    # Build the main prompt
    prompt_parts = [
        "Please extract the following fields from the document:",
        "",
        fields_formatted,
        "",
        "Return your response as a JSON object (or array of objects if multiple records exist).",
        "Use the exact field names provided above as keys.",
        ""
    ]
    
    # Add document type hint if provided
    if document_type:
        prompt_parts.insert(0, f"Document Type: {document_type}\n")
    
    # Add PDF text content if provided (for text-only mode)
    if pdf_text:
        prompt_parts.extend([
            "--- DOCUMENT CONTENT ---",
            pdf_text,
            "--- END DOCUMENT ---",
            ""
        ])
    
    # Add custom prompt if provided
    if custom_prompt:
        prompt_parts.extend([
            "Additional Instructions:",
            custom_prompt,
            ""
        ])
    
    # Add output format reminder
    prompt_parts.extend([
        "Respond with ONLY valid JSON. Example format:",
        '{"field1": "value1", "field2": "value2"} or',
        '[{"field1": "value1"}, {"field1": "value2"}] for multiple records'
    ])
    
    return "\n".join(prompt_parts)


def build_vision_messages(
    fields: List[str],
    images: List,
    custom_prompt: Optional[str] = None,
    pdf_text: Optional[str] = None
) -> List[dict]:
    """
    Build messages for vision-language model with images.

    Args:
        fields: List of field names to extract
        images: List of PIL Images or image paths
        custom_prompt: Optional additional instructions
        pdf_text: Optional extracted text to supplement vision

    Returns:
        List of message dictionaries for the model

    Note:
        Qwen2-VL requires ALL message content to be a list of dicts with "type" key.
        Even system messages must use: [{"type": "text", "text": "..."}]
    """
    # Build content list with images first
    content = []

    # Add images
    for img in images:
        content.append({
            "type": "image",
            "image": img
        })

    # Build text prompt
    text_prompt = build_extraction_prompt(
        fields=fields,
        pdf_text=pdf_text,
        custom_prompt=custom_prompt
    )

    content.append({
        "type": "text",
        "text": text_prompt
    })

    # Qwen2-VL requires content to be a list of dicts for ALL messages
    # System message also needs [{"type": "text", "text": "..."}] format
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": DEFAULT_SYSTEM_PROMPT}]
        },
        {
            "role": "user",
            "content": content
        }
    ]

    return messages


def build_text_only_messages(
    fields: List[str],
    pdf_text: str,
    custom_prompt: Optional[str] = None
) -> List[dict]:
    """
    Build messages for text-only extraction (no images).

    Args:
        fields: List of field names to extract
        pdf_text: Extracted text from PDF
        custom_prompt: Optional additional instructions

    Returns:
        List of message dictionaries for the model

    Note:
        Qwen2-VL requires ALL message content to be a list of dicts with "type" key.
    """
    text_prompt = build_extraction_prompt(
        fields=fields,
        pdf_text=pdf_text,
        custom_prompt=custom_prompt
    )

    # Qwen2-VL requires content to be a list of dicts for ALL messages
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": DEFAULT_SYSTEM_PROMPT}]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": text_prompt}]
        }
    ]

    return messages

