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
    # Try to extract JSON from the response
    response = response.strip()
    
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
        
        # Handle both single object and list of objects
        if isinstance(parsed, dict):
            # Check if it's wrapped in a key like "data" or "rows"
            for key in ["data", "rows", "records", "items", "results"]:
                if key in parsed and isinstance(parsed[key], list):
                    return parsed[key]
            return [parsed]
        elif isinstance(parsed, list):
            return parsed
        else:
            return [{"raw_response": str(parsed)}]
            
    except json.JSONDecodeError:
        # If JSON parsing fails, return raw response
        return [{"raw_response": response}]

