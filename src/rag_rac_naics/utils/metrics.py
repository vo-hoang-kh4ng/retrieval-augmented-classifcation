# Utility functions
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

def setup_logger(log_level: str = "INFO") -> logging.Logger:
    """Setup and return a standard Python logger.

    This avoids a hard dependency on loguru while providing configurable
    log level control.
    """
    level = getattr(logging, (log_level or "INFO").upper(), logging.INFO)
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger("rag_rac_naics")

def load_naics_data(file_path: str) -> List[Dict[str, Any]]:
    """Load NAICS codes from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_dbpedia_data(file_path: str) -> List[Dict[str, Any]]:
    """Load DBPedia dataset"""
    # Placeholder implementation: expect JSON list
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def calculate_accuracy_at_k(predictions: List[List[str]], ground_truth: List[str], k: int = 5) -> float:
    """Calculate accuracy@k given list of top-k predictions per example.

    Args:
        predictions: List of lists where each inner list is ordered top-N labels.
        ground_truth: List of gold labels, same length as predictions.
        k: Consider top-k for correctness.
    """
    if not predictions:
        return 0.0
    correct = 0
    for topn, truth in zip(predictions, ground_truth):
        if truth in (topn[:k] if isinstance(topn, list) else [topn]):
            correct += 1
    return correct / max(1, len(predictions))

def calculate_fuzzy_accuracy(predictions: List[str], ground_truth: List[str]) -> float:
    """Calculate fuzzy accuracy for hierarchical (e.g., NAICS) codes.

    A prediction is considered correct if it matches the ground-truth by
    full code, or by prefix of 2/3/4 digits (weighing all equally here).
    """
    if not predictions:
        return 0.0
    def match(pred: str, gold: str) -> bool:
        p = (pred or "").strip()
        g = (gold or "").strip()
        if not p or not g:
            return False
        if p == g:
            return True
        # Prefix levels (common NAICS hierarchy)
        for L in (2, 3, 4):
            if len(p) >= L and len(g) >= L and p[:L] == g[:L]:
                return True
        return False

    correct = sum(1 for p, g in zip(predictions, ground_truth) if match(p, g))
    return correct / max(1, len(predictions))
