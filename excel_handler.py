"""Excel template handling module."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

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
    output_data = []
    for row_data in extracted_data:
        aligned_row = {}
        for header in headers:
            # Try exact match first, then case-insensitive
            value = row_data.get(header)
            if value is None:
                # Case-insensitive lookup
                for key in row_data:
                    if key.lower() == header.lower():
                        value = row_data[key]
                        break
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
    response = response.strip()
    original_response = response  # Keep for fallback

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
        rows = _normalize_parsed_data(parsed, headers)
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

        # If JSON parsing fails, return raw response
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
        for key in ["data", "rows", "records", "items", "results", "entries"]:
            if key in parsed and isinstance(parsed[key], list):
                return _normalize_parsed_data(parsed[key], headers)

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

        if has_array_values and max_len > 1:
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

        # Single object - return as single row
        return [parsed]

    # Handle list
    elif isinstance(parsed, list):
        if len(parsed) == 0:
            return []

        # Check if it's a list of objects
        if all(isinstance(item, dict) for item in parsed):
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

