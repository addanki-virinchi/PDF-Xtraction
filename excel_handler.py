"""Excel template handling module."""

import json
import re
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Iterable, Set

import pandas as pd
from openpyxl import load_workbook, Workbook
from openpyxl.utils.dataframe import dataframe_to_rows


def read_excel_headers(excel_path: Union[str, Path]) -> List[str]:
    """
    Read column headers from the first sheet of an Excel file.
    
    Args:
        excel_path: Path to the Excel template file
        
    Returns:
        List of column header names
    """
    df = pd.read_excel(excel_path, sheet_name=0, nrows=0)
    return df.columns.tolist()


def read_excel_template(excel_path: Union[str, Path]) -> pd.DataFrame:
    """
    Read the entire Excel template including any existing data.
    
    Args:
        excel_path: Path to the Excel template file
        
    Returns:
        DataFrame with template data
    """
    return pd.read_excel(excel_path, sheet_name=0)


def create_output_excel(
    template_path: Union[str, Path],
    output_path: Union[str, Path],
    extracted_data: List[Dict[str, Any]]
) -> Path:
    """
    Create output Excel file with extracted data.
    
    Args:
        template_path: Path to the original Excel template
        output_path: Path for the output Excel file
        extracted_data: List of dictionaries containing extracted data
        
    Returns:
        Path to the created output file
    """
    output_path = Path(output_path)
    
    # Read template to get headers and any formatting
    template_df = read_excel_template(template_path)
    headers = template_df.columns.tolist()
    
    # Create DataFrame from extracted data, aligning with template columns
    def _normalize_key(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", value.lower())

    output_data = []
    for row_data in extracted_data:
        aligned_row = {}
        normalized_row_keys = {
            _normalize_key(k): k for k in row_data.keys()
            if isinstance(k, str)
        }
        for header in headers:
            # Try exact match first, then case-insensitive
            value = row_data.get(header)
            if value is None:
                # Case-insensitive lookup
                for key in row_data:
                    if key.lower() == header.lower():
                        value = row_data[key]
                        break
            if value is None:
                # Fallback to normalized key matching (e.g., Sold-to vs Sold_to)
                normalized_header = _normalize_key(header)
                matched_key = normalized_row_keys.get(normalized_header)
                if matched_key is not None:
                    value = row_data.get(matched_key)
            aligned_row[header] = value if value is not None else ""
        output_data.append(aligned_row)
    
    # Create output DataFrame
    output_df = pd.DataFrame(output_data, columns=headers)
    
    # Write to Excel
    output_df.to_excel(output_path, index=False, sheet_name="Extracted Data")
    
    return output_path


def parse_model_response_to_rows(
    response: str,
    headers: List[str]
) -> List[Dict[str, Any]]:
    """
    Parse model response text into structured row data.

    The model is expected to return JSON format. This function handles
    various response formats and normalizes them.

    Args:
        response: Raw model response text
        headers: Expected column headers

    Returns:
        List of dictionaries, each representing a row
    """
    import logging
    logger = logging.getLogger(__name__)

    # Try to extract JSON from the response
    if response is None:
        logger.warning("Model response is None")
        return []

    response = response.strip()
    original_response = response  # Keep for fallback
    if not original_response:
        logger.warning("Model response is empty")
        return []

    if os.getenv("DEBUG_MODEL_RESPONSE") == "1":
        snippet = original_response[:1000]
        logger.info("Model response snippet (first 1000 chars): %s", snippet)

    # Handle markdown code blocks
    if "```json" in response:
        start = response.find("```json") + 7
        end = response.find("```", start)
        if end > start:
            response = response[start:end].strip()
    elif "```" in response:
        start = response.find("```") + 3
        end = response.find("```", start)
        if end > start:
            response = response[start:end].strip()

    try:
        parsed = json.loads(response)
        if isinstance(parsed, str):
            parsed_str = parsed.strip()
            if (parsed_str.startswith("{") and parsed_str.endswith("}")) or (
                parsed_str.startswith("[") and parsed_str.endswith("]")
            ):
                try:
                    parsed = json.loads(parsed_str)
                except json.JSONDecodeError:
                    pass
        rows = _normalize_parsed_data(parsed, headers)
        if not rows:
            logger.warning("Parsed response but produced 0 rows")
        logger.info(f"Parsed {len(rows)} rows from model response")
        return rows

    except json.JSONDecodeError as e:
        logger.warning(f"JSON parsing failed: {e}")
        # Try to find JSON within the response
        json_match = _extract_json_from_text(original_response)
        if json_match:
            try:
                parsed = json.loads(json_match)
                rows = _normalize_parsed_data(parsed, headers)
                logger.info(f"Extracted and parsed {len(rows)} rows from embedded JSON")
                return rows
            except json.JSONDecodeError:
                pass

        # Try simple key:value parsing as a fallback
        parsed_kv = _parse_key_value_lines(original_response)
        if parsed_kv:
            logger.warning("Parsed response using key:value fallback")
            return [parsed_kv]

        # If parsing fails, return raw response
        logger.warning("Could not parse response as JSON, returning raw")
        return [{"raw_response": original_response}]


def _extract_json_from_text(text: str) -> Optional[str]:
    """Try to extract JSON object or array from text."""
    import re

    # Try to find a JSON array
    array_match = re.search(r'\[[\s\S]*\]', text)
    if array_match:
        return array_match.group()

    # Try to find a JSON object
    obj_match = re.search(r'\{[\s\S]*\}', text)
    if obj_match:
        return obj_match.group()

    return None


def _parse_key_value_lines(text: str) -> Dict[str, Any]:
    """Parse simple 'Key: Value' lines into a dict."""
    rows = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key:
            rows[key] = value
    return rows


def _normalize_parsed_data(parsed: Any, headers: List[str]) -> List[Dict[str, Any]]:
    """
    Normalize parsed JSON data into a list of row dictionaries.

    Handles various formats:
    - Single object: {"field1": "val1", "field2": "val2"}
    - List of objects: [{"field1": "val1"}, {"field1": "val2"}]
    - Wrapped in a key: {"data": [...], "rows": [...]}
    - Object with array values: {"field1": ["val1", "val2"], "field2": ["val3", "val4"]}
    """
    # Handle dict
    if isinstance(parsed, dict):
        # Check if it's wrapped in a key like "data" or "rows"
        wrapper_keys = [
            "data", "rows", "records", "items", "results", "entries",
            "output", "result", "response", "extracted_data"
        ]
        normalized_to_original = {_normalize_label(k): k for k in parsed.keys()}
        for key in wrapper_keys:
            original_key = normalized_to_original.get(_normalize_label(key))
            if original_key and isinstance(parsed[original_key], (list, dict)):
                return _normalize_parsed_data(parsed[original_key], headers)

        # Handle key/value pair list under a common wrapper
        for key, value in parsed.items():
            if isinstance(value, list) and _looks_like_kv_pairs(value):
                row = _kv_pairs_to_row(value)
                return [row] if row else []

        # Check if values are arrays (columnar format) - need to transpose
        # e.g., {"Name": ["Alice", "Bob"], "Age": [25, 30]}
        array_values = {}
        max_len = 0
        has_array_values = False

        for key, value in parsed.items():
            if isinstance(value, list) and len(value) > 0:
                has_array_values = True
                array_values[key] = value
                max_len = max(max_len, len(value))
            else:
                array_values[key] = value

        if has_array_values:
            if max_len > 1:
                # Transpose columnar data to rows
                rows = []
                for i in range(max_len):
                    row = {}
                    for key, value in array_values.items():
                        if isinstance(value, list):
                            row[key] = value[i] if i < len(value) else None
                        else:
                            row[key] = value  # Repeat scalar values
                    rows.append(row)
                return rows
            # Flatten single-item lists into scalars
            flattened = {}
            for key, value in array_values.items():
                if isinstance(value, list):
                    flattened[key] = value[0] if value else None
                else:
                    flattened[key] = value
            return [flattened]

        # Single object - return as single row
        return [parsed]

    # Handle list
    elif isinstance(parsed, list):
        if len(parsed) == 0:
            return []

        # Check if it's a list of objects
        if all(isinstance(item, dict) for item in parsed):
            if _looks_like_kv_pairs(parsed):
                row = _kv_pairs_to_row(parsed)
                return [row] if row else []
            return parsed

        # Check if it's a list of lists (table format without headers)
        # e.g., [["val1", "val2"], ["val3", "val4"]]
        if all(isinstance(item, list) for item in parsed):
            rows = []
            for row_values in parsed:
                row = {}
                for i, value in enumerate(row_values):
                    if i < len(headers):
                        row[headers[i]] = value
                    else:
                        row[f"column_{i+1}"] = value
                rows.append(row)
            return rows

        # List of primitives - might be values for a single column
        # Return each as a separate row with the first header
        if headers:
            return [{headers[0]: item} for item in parsed]
        return [{"value": item} for item in parsed]

    # Other types - wrap in a dict
    return [{"raw_value": str(parsed)}]


def _looks_like_kv_pairs(items: List[Dict[str, Any]]) -> bool:
    """Detect list-of-dict key/value pairs like [{'field': 'X', 'value': 'Y'}]."""
    if not items:
        return False
    key_fields = _normalize_labels((
        "field", "name", "key", "label", "column", "header", "question",
        "field_name", "fieldname"
    ))
    value_fields = _normalize_labels((
        "value", "val", "answer", "text", "content", "field_value", "fieldvalue"
    ))
    for item in items:
        if not isinstance(item, dict):
            return False
        normalized_keys = {_normalize_label(k) for k in item.keys()}
        if not normalized_keys.intersection(key_fields):
            return False
        if not normalized_keys.intersection(value_fields):
            return False
    return True


def _kv_pairs_to_row(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Convert list of key/value dicts into a single row dict."""
    key_fields = _normalize_labels((
        "field", "name", "key", "label", "column", "header", "question",
        "field_name", "fieldname"
    ))
    value_fields = _normalize_labels((
        "value", "val", "answer", "text", "content", "field_value", "fieldvalue"
    ))
    row: Dict[str, Any] = {}
    for item in items:
        key = _extract_first_value(item, key_fields)
        if not key:
            continue
        value = _extract_first_value(item, value_fields, default=None)
        row[key] = value
    return row


def _normalize_label(label: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(label).lower())


def _normalize_labels(labels: Iterable[str]) -> Set[str]:
    return {_normalize_label(label) for label in labels}


def _extract_first_value(item: Dict[str, Any], normalized_keys: Set[str], default: Any = "") -> Any:
    for key, value in item.items():
        if _normalize_label(key) in normalized_keys and value is not None:
            return str(value).strip() if isinstance(value, str) else value
    return default
