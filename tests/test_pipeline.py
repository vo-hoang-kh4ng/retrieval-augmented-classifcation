"""End-to-end tests for RAG RAC NAICS pipeline that avoid external APIs.

These tests rely on the hashing-based fallback embeddings and temporary
ChromaDB directories, so they are safe and deterministic in CI.
"""

import os
import shutil
import tempfile

import numpy as np
import pytest

from src.rag_rac_naics.clients import LLMClients
from src.rag_rac_naics.classifiers.domain_classifier import DomainClassifier
from src.rag_rac_naics.classifiers.rac_classifier import RACClassifier
from src.rag_rac_naics.retrieval.vector_store import VectorStore
from src.rag_rac_naics.embeddings.naics_embeddings import NAICSEmbeddings
from src.rag_rac_naics.retrieval.naics_retriever import NAICSRetriever
from src.rag_rac_naics.utils.metrics import (
    calculate_accuracy_at_k,
    calculate_fuzzy_accuracy,
)


@pytest.fixture(autouse=True)
def no_api_keys(monkeypatch):
    # Ensure tests do not hit external APIs
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "similarity_only")
    # Lower embedding dim to speed up
    monkeypatch.setenv("EMBEDDING_DIM", "64")


def test_domain_classifier_heuristic():
    clients = LLMClients.from_env()
    clf = DomainClassifier(clients)
    assert clf.classify("tell me a joke") == "OUT_OF_DOMAIN"
    assert clf.classify("account login issue") == "IN_DOMAIN"


def test_rac_end_to_end():
    tmpdir = tempfile.mkdtemp()
    try:
        clients = LLMClients.from_env()
        store = VectorStore(db_path=os.path.join(tmpdir, "rac_chroma"), collection_name="rac")
        rac = RACClassifier(clients, store, k=3)

        texts = [
            "Software development and programming services",
            "Web design and development",
            "Mobile app development",
            "Restaurant and food service",
            "Fast food restaurant",
        ]
        labels = ["541511", "541511", "541511", "722513", "722513"]

        rac.fit(texts, labels)
        result = rac.classify_with_retrieval("Custom web application development", use_llm=False)
        assert result["prediction"] in set(labels)
        assert result["confidence"] >= 0.0
        assert len(result["retrieved_examples"]) > 0
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_naics_shortlist_and_similarity_fallback():
    tmpdir = tempfile.mkdtemp()
    try:
        clients = LLMClients.from_env()
        embeddings = NAICSEmbeddings(clients)
        store = VectorStore(db_path=os.path.join(tmpdir, "naics_chroma"), collection_name="naics")
        retriever = NAICSRetriever(store, embeddings)

        texts = [
            "Software development and programming services",
            "Fast food restaurant",
            "Coffee shop and cafe services",
        ]
        labels = ["541511", "722513", "722515"]
        vecs = embeddings.get_embeddings(texts)
        store.add_texts(texts, vecs, labels)

        shortlist = retriever.shortlist_codes("We build websites and apps", k=2)
        assert 1 <= len(shortlist) <= 2

        result = retriever._fallback_to_similarity(shortlist)
        assert result["prediction"] in {"541511", "722513", "722515"}
        assert result["from"] == "similarity"
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_metrics_functions():
    preds_topk = [["A", "B", "C"], ["B", "A", "C"], ["C", "A", "B"]]
    gold = ["A", "A", "B"]

    acc1 = calculate_accuracy_at_k(preds_topk, gold, k=1)
    acc2 = calculate_accuracy_at_k(preds_topk, gold, k=2)
    assert 0.0 <= acc1 <= 1.0
    assert 0.0 <= acc2 <= 1.0
    assert acc2 >= acc1

    # Fuzzy accuracy tests for hierarchical codes
    fuzzy = calculate_fuzzy_accuracy(["541511", "7225", "4481"], ["541511", "722513", "448140"])
    assert 0.0 <= fuzzy <= 1.0
