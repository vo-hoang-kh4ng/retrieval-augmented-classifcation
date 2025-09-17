"""NAICS Retriever for industry classification.

Shortlist using vector similarity, then optionally ask an LLM to pick the
final NAICS code from the shortlist (multiple-choice style).
"""

import numpy as np
from typing import List, Dict, Any, Tuple
import os
import requests
from .vector_store import VectorStore
from ..embeddings.naics_embeddings import NAICSEmbeddings

class NAICSRetriever:
    def __init__(self, vector_store: VectorStore, embeddings: NAICSEmbeddings):
        self.vector_store = vector_store
        self.embeddings = embeddings
    
    def shortlist_codes(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Shortlist NAICS codes based on similarity"""
        # Get query embedding
        query_embedding = self.embeddings.get_embeddings([query])[0]
        
        # Search for similar codes
        texts, labels, scores = self.vector_store.search(query_embedding, k=k)
        
        results = []
        for text, label, score in zip(texts, labels, scores):
            results.append({
                "code": label,
                "description": text,
                "score": score,
                "confidence": self._calculate_confidence(score)
            })
        
        return results
    
    def classify_with_llm(self, query: str, shortlist: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Use LLM to select final NAICS code from shortlist"""
        if not shortlist:
            return {"prediction": "UNKNOWN", "reason": "empty_shortlist"}

        api_key = os.getenv("OPENAI_API_KEY")
        model = os.getenv("LLM_MODEL", "gpt-3.5-turbo")

        # Build options text
        options_lines = []
        for i, item in enumerate(shortlist, start=1):
            options_lines.append(f"{i}. {item['code']} - {item['description']}")
        options_text = "\n".join(options_lines)

        prompt = (
            "You are a NAICS code classifier. Select the most appropriate NAICS code from the given options.\n\n"
            f"Query: {query}\n\nOptions:\n{options_text}\n\n"
            "Respond with the exact NAICS code only, or UNKNOWN if none match well."
        )

        if api_key:
            try:
                url = "https://api.openai.com/v1/chat/completions"
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                }
                data = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": "Return only a NAICS code or UNKNOWN."},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": float(os.getenv("TEMPERATURE", "0.0")),
                    "max_tokens": 12,
                }
                resp = requests.post(url, headers=headers, json=data, timeout=30)
                resp.raise_for_status()
                out = resp.json()
                answer = (
                    out.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                    .strip()
                )
                # Normalize to a code from shortlist or UNKNOWN
                codes = [str(item["code"]).strip() for item in shortlist]
                for code in codes:
                    if code in answer:
                        chosen = next(x for x in shortlist if str(x["code"]).strip() == code)
                        return {"prediction": code, "from": "llm", "chosen": chosen}
                if "unknown" in answer.lower():
                    return {"prediction": "UNKNOWN", "from": "llm"}
            except Exception:
                # Fall through to similarity top-1
                pass

        # Fallback: choose top by score
        best = max(shortlist, key=lambda x: x.get("score", 0.0))
        return {"prediction": best["code"], "from": "similarity", "chosen": best}
    
    def _calculate_confidence(self, score: float) -> float:
        """Calculate confidence based on similarity score"""
        return min(score * 1.5, 1.0)  # Scale and cap at 1.0
