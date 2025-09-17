"""RAC (Retrieval-Augmented Classification) Core.

Embeddings strategy:
- Try OpenAI embeddings via REST when OPENAI_API_KEY is present
- Fallback to a deterministic hashing-based embedding (no external deps)

LLM strategy:
- Try OpenAI Chat Completions for few-shot classification
- Fallback to KNN majority from retrieved labels
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import os
import requests
from ..clients import LLMClients
from ..retrieval.vector_store import VectorStore
from .knn_classifier import KNNClassifier

class RACClassifier:
    def __init__(self, clients: LLMClients, vector_store: VectorStore, k: int = 5):
        self.clients = clients
        self.vector_store = vector_store
        self.k = k
        self.knn = KNNClassifier(k=k)
    
    def fit(self, texts: List[str], labels: List[str]) -> None:
        """Fit RAC on training data"""
        # Get embeddings for training texts
        embeddings = self._get_embeddings(texts)
        
        # Store in vector store
        self.vector_store.add_texts(texts, embeddings, labels)
        
        # Fit KNN
        self.knn.fit(embeddings, labels)
    
    def classify_with_retrieval(self, text: str, use_llm: bool = True) -> Dict[str, Any]:
        """Classify text using retrieval + LLM"""
        # Get query embedding
        query_embedding = self._get_embeddings([text])[0]
        
        # Retrieve similar examples
        similar_texts, similar_labels, scores = self.vector_store.search(
            query_embedding, k=self.k
        )
        
        if use_llm:
            # Use LLM to classify based on retrieved examples
            prediction = self._llm_classify(text, similar_texts, similar_labels)
            confidence = self._calculate_confidence(scores)
        else:
            # Use KNN mode
            prediction = self.knn.predict([query_embedding])[0]
            confidence = 1.0
        
        return {
            "prediction": prediction,
            "confidence": confidence,
            "retrieved_examples": list(zip(similar_texts, similar_labels, scores)),
            "method": "llm" if use_llm else "knn"
        }
    
    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for texts"""
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
                # Fall through to hashing fallback
                pass

        # Fallback: simple hashing-based bag-of-words embedding
        dim = int(os.getenv("EMBEDDING_DIM", "384"))
        vectors: List[List[float]] = []
        for t in texts:
            vec = [0.0] * dim
            for token in (t or "").lower().split():
                idx = (hash(token) % dim)
                vec[idx] += 1.0
            # L2 normalize
            norm = np.linalg.norm(vec) or 1.0
            vectors.append([v / norm for v in vec])
        return np.array(vectors, dtype=float)
    
    def _llm_classify(self, text: str, examples: List[str], labels: List[str]) -> str:
        """Use LLM to classify based on retrieved examples"""
        # Build prompt
        prompt = self._build_rac_prompt(text, examples, labels)
        api_key = self.clients.openai_api_key
        model = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
        unique_labels = list(dict.fromkeys(labels)) or []

        if api_key and unique_labels:
            try:
                url = "https://api.openai.com/v1/chat/completions"
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                }
                data = {
                    "model": model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "Return only one of the provided classes or UNKNOWN.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": float(os.getenv("TEMPERATURE", "0.0")),
                    "max_tokens": 8,
                }
                resp = requests.post(url, headers=headers, json=data, timeout=30)
                resp.raise_for_status()
                out = resp.json()
                text_out = (
                    out.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                    .strip()
                )
                # Normalize to one of the labels or UNKNOWN
                normalized = self._normalize_label(text_out, unique_labels)
                return normalized
            except Exception:
                # Fall back to majority vote
                pass

        # Fallback: majority label among retrieved labels
        if labels:
            return max(set(labels), key=labels.count)
        return "UNKNOWN"

    def _build_rac_prompt(self, text: str, examples: List[str], labels: List[str]) -> str:
        # Try reading config/prompts.yaml rac_classification block
        template = (
            "You are a classifier. Use the following examples to classify the input text.\n\n"
            f"Classes: {sorted(list(set(labels)))}\n\n"
            "Examples:\n" +
            "\n".join(f"- [{lbl}] {ex}" for ex, lbl in list(zip(examples, labels))[: self.k]) +
            "\n\nInput: " + text + "\n\nRespond with the single best class label or UNKNOWN."
        )
        # Optionally replace with prompts.yaml content if present
        try:
            path = "config/prompts.yaml"
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                marker = "rac_classification: |"
                if marker in content:
                    after = content.split(marker, 1)[1]
                    lines = []
                    for line in after.splitlines():
                        if line.startswith(" ") or line.startswith("\t") or line.strip() == "":
                            lines.append(line.lstrip())
                        else:
                            break
                    yaml_template = "\n".join(lines).strip()
                    if yaml_template:
                        classes_str = ", ".join(sorted(list(set(labels))))
                        examples_str = "\n".join(
                            f"- [{lbl}] {ex}" for ex, lbl in list(zip(examples, labels))[: self.k]
                        )
                        return (
                            yaml_template
                            .replace("{classes}", classes_str)
                            .replace("{examples}", examples_str)
                            .replace("{input}", text)
                        )
        except Exception:
            pass
        return template

    def _normalize_label(self, output_text: str, allowed_labels: List[str]) -> str:
        out = (output_text or "").strip()
        if not out:
            return "UNKNOWN"
        # Exact or case-insensitive match
        for lab in allowed_labels:
            if out == lab or out.lower() == lab.lower():
                return lab
        # Try to find first occurrence of any label as substring
        lower = out.lower()
        for lab in allowed_labels:
            if lab.lower() in lower:
                return lab
        if "unknown" in lower:
            return "UNKNOWN"
        return "UNKNOWN"
    
    def _calculate_confidence(self, scores: List[float]) -> float:
        """Calculate confidence based on retrieval scores"""
        return float(np.mean(scores))
