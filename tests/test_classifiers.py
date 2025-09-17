# Test files
import pytest
import tempfile
import shutil
import os
import numpy as np

from src.rag_rac_naics.classifiers.domain_classifier import DomainClassifier
from src.rag_rac_naics.classifiers.knn_classifier import KNNClassifier
from src.rag_rac_naics.classifiers.rac_classifier import RACClassifier
from src.rag_rac_naics.retrieval.vector_store import VectorStore
from src.rag_rac_naics.utils.metrics import calculate_accuracy_at_k
from src.rag_rac_naics.clients import LLMClients


@pytest.fixture
def no_api_keys(monkeypatch):
    # Ensure tests do not hit external APIs
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "similarity_only")


def test_domain_classifier(no_api_keys):
    """Test domain classification with heuristic fallback"""
    clients = LLMClients.from_env()
    clf = DomainClassifier(clients)
    assert clf.classify("tell me a joke") == "OUT_OF_DOMAIN"
    assert clf.classify("account login issue") == "IN_DOMAIN"


def test_knn_classifier():
    """Test KNN classification"""
    # TODO: Implement test
    pass

def test_rac_classifier():
    """Test RAC classification"""
    # TODO: Implement test
    pass

def test_vector_store():
    """Test vector store operations"""
    # TODO: Implement test
    pass

def test_metrics():
    """Test metrics calculation"""
    # TODO: Implement test
    pass
