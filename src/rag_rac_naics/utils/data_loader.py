"""Data loaders for NAICS and other datasets."""

import json
from typing import List, Dict, Any


def load_naics_json(file_path: str) -> List[Dict[str, Any]]:
    """Load NAICS codes from a JSON file.

    Expected format: list of objects with keys like
    {"code": "311111", "title": "Example", "description": "..."}
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("NAICS JSON must be a list of objects")
    return data


