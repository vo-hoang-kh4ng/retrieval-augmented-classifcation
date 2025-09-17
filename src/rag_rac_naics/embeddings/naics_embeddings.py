"""Embeddings for NAICS classification.

Preference order:
- OpenAI embeddings via REST (if OPENAI_API_KEY present)
- Hashing-based fallback (no external deps)
"""

import numpy as np
from typing import List
import os
import requests
from ..clients import LLMClients


class NAICSEmbeddings:
    def __init__(self, clients: LLMClients):
        self.clients = clients

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for NAICS texts."""
        api_key = self.clients.openai_api_key
        model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        if api_key:
            try:
                url = "https://api.openai.com/v1/embeddings"
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                }
                data = {"model": model, "input": texts}
                resp = requests.post(url, headers=headers, json=data, timeout=30)
                resp.raise_for_status()
                out = resp.json()
                vectors = [item["embedding"] for item in out.get("data", [])]
                if vectors:
                    return np.array(vectors, dtype=float)
            except Exception:
                pass

        # Fallback hashing embedding
        dim = int(os.getenv("EMBEDDING_DIM", "384"))
        vectors: List[List[float]] = []
        for t in texts:
            vec = [0.0] * dim
            for token in (t or "").lower().split():
                idx = (hash(token) % dim)
                vec[idx] += 1.0
            norm = np.linalg.norm(vec) or 1.0
            vectors.append([v / norm for v in vec])
        return np.array(vectors, dtype=float)

    def get_phrase_embeddings(self, phrases: List[str]) -> np.ndarray:
        """Get embeddings for short phrases (NAICS descriptions)."""
        return self.get_embeddings(phrases)
