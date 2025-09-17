"""NAICS Retriever for industry classification.

Shortlist using vector similarity, then optionally ask an LLM to pick the
final NAICS code from the shortlist (multiple-choice style).
"""
import os
import logging
import requests
import numpy as np
from typing import List, Dict, Any, Tuple

# Sử dụng thư viện chính thức của OpenAI để ổn định hơn
import openai

from .vector_store import VectorStore
from ..embeddings.naics_embeddings import NAICSEmbeddings

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class NAICSRetriever:
    """
    Retrieves and classifies NAICS codes by first shortlisting candidates
    using vector similarity and then using an LLM for final selection.
    """
    def __init__(self, vector_store: VectorStore, embeddings: NAICSEmbeddings):
        """
        Initializes the retriever.
        """
        self.vector_store = vector_store
        self.embeddings = embeddings

        # Cải tiến: Khởi tạo các client dựa trên biến môi trường
        self.llm_provider = os.getenv("LLM_PROVIDER", "similarity_only").lower()
        self.openai_client = None
        self.gemini_api_key = None

        if self.llm_provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self.openai_client = openai.OpenAI(api_key=api_key)
            else:
                logging.warning("LLM_PROVIDER is 'openai' but OPENAI_API_KEY is not set.")
        elif self.llm_provider == "gemini":
            self.gemini_api_key = os.getenv("GEMINI_API_KEY")
            if not self.gemini_api_key:
                logging.warning("LLM_PROVIDER is 'gemini' but GEMINI_API_KEY is not set.")

    def shortlist_codes(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """
        Shortlists NAICS codes based on semantic similarity to the query.
        """
        logging.info(f"Shortlisting top {k} codes for query: '{query[:50]}...'")
        query_embedding = self.embeddings.get_embeddings([query])[0]
        texts, metadatas, scores = self.vector_store.search(query_embedding, k=k)

        results = []
        for text, metadata, score in zip(texts, metadatas, scores):
            results.append({
                "code": metadata.get("label", "N/A"),
                "description": text,
                "score": score
            })
        
        logging.info(f"Found {len(results)} candidates in shortlist.")
        return results

    def classify(self, query: str, shortlist: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Selects the final NAICS code from the shortlist using the configured LLM provider.
        """
        if not shortlist:
            logging.warning("Shortlist is empty. Cannot classify.")
            return {"prediction": "UNKNOWN", "reason": "empty_shortlist"}

        if self.llm_provider == "openai" and self.openai_client:
            return self._classify_with_openai(query, shortlist)
        elif self.llm_provider == "gemini" and self.gemini_api_key:
            return self._classify_with_gemini(query, shortlist)
        else:
            logging.warning(f"LLM provider '{self.llm_provider}' not configured correctly. Falling back to similarity.")
            return self._fallback_to_similarity(shortlist)

    def _build_prompt(self, query: str, shortlist: List[Dict[str, Any]]) -> str:
        """Helper to build the common prompt for LLMs."""
        options_lines = [f"{i}. {item['code']} - {item['description']}" for i, item in enumerate(shortlist, start=1)]
        options_text = "\n".join(options_lines)

        return (
            "You are an expert economic analyst specializing in the NAICS classification system. "
            "Your task is to select the single most appropriate NAICS code from the given options for the user's query.\n\n"
            f"User Query: \"{query}\"\n\n"
            f"Options:\n{options_text}\n\n"
            "Respond with only the 6-digit NAICS code for your choice. Do not add any explanation. "
            "If none of the options are a good fit, respond with the word UNKNOWN."
        )

    def _classify_with_openai(self, query: str, shortlist: List[Dict[str, Any]]) -> Dict[str, Any]:
        model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        prompt = self._build_prompt(query, shortlist)
        try:
            logging.info(f"Querying OpenAI model ({model}) for final classification...")
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful NAICS classification assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=15,
            )
            answer = response.choices[0].message.content.strip()
            return self._process_llm_answer(answer, shortlist, "openai")
        except Exception as e:
            logging.error(f"OpenAI API error: {e}. Falling back to similarity.")
            return self._fallback_to_similarity(shortlist)

    def _classify_with_gemini(self, query: str, shortlist: List[Dict[str, Any]]) -> Dict[str, Any]:
        model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-preview-05-20")
        prompt = self._build_prompt(query, shortlist)
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self.gemini_api_key}"
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        try:
            logging.info(f"Querying Google Gemini model ({model}) for final classification...")
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            answer = result['candidates'][0]['content']['parts'][0]['text'].strip()
            return self._process_llm_answer(answer, shortlist, "gemini")
        except Exception as e:
            logging.error(f"Google Gemini API error: {e}. Falling back to similarity.")
            return self._fallback_to_similarity(shortlist)

    def _process_llm_answer(self, answer: str, shortlist: List[Dict[str, Any]], provider: str) -> Dict[str, Any]:
        """Processes and validates the answer from an LLM."""
        logging.info(f"LLM ({provider}) response: '{answer}'")
        shortlist_codes = {str(item["code"]).strip() for item in shortlist}
        if answer in shortlist_codes:
            chosen = next(item for item in shortlist if str(item["code"]).strip() == answer)
            return {"prediction": answer, "from": provider, "chosen": chosen}
        
        logging.warning(f"LLM response '{answer}' not in shortlist. Falling back to similarity.")
        return self._fallback_to_similarity(shortlist)

    def _fallback_to_similarity(self, shortlist: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Selects the best option based on the initial similarity score."""
        logging.info("Falling back to top similarity score.")
        if not shortlist: return {"prediction": "UNKNOWN", "from": "similarity", "chosen": None}
        best_match = max(shortlist, key=lambda x: x.get("score", 0.0))
        return {"prediction": best_match["code"], "from": "similarity", "chosen": best_match}

