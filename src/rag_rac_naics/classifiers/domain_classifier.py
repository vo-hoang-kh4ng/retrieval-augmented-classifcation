"""Domain Classifier - Zero-shot LLM filter before retrieval.

Implements two modes:
- OpenAI Chat Completions (if OPENAI_API_KEY is set)
- Heuristic fallback (keyword-based)

It optionally loads a prompt from config/prompts.yaml; otherwise,
it uses a sensible default prompt.
"""

from typing import Literal, Optional
from ..clients import LLMClients
import os
import requests


class DomainClassifier:
    def __init__(self, clients: LLMClients, prompts_file_path: str = "config/prompts.yaml"):
        self.clients = clients
        self.prompts_file_path = prompts_file_path

    def classify(self, query: str) -> Literal["IN_DOMAIN", "OUT_OF_DOMAIN"]:
        """Classify if query is in-domain before retrieval.

        Attempts OpenAI first; falls back to heuristic when unavailable or on error.
        """
        query = (query or "").strip()
        if not query:
            return "OUT_OF_DOMAIN"

        openai_key = self.clients.openai_api_key
        if openai_key:
            try:
                prompt = self._build_prompt(query)
                label = self._classify_with_openai(prompt, openai_key)
                if label in ("IN_DOMAIN", "OUT_OF_DOMAIN"):
                    return label  # type: ignore
            except Exception:
                # Silent fallback to heuristic
                pass

        return self._heuristic_classify(query)

    def is_in_domain(self, query: str) -> bool:
        """Convenience method."""
        return self.classify(query) == "IN_DOMAIN"

    def _build_prompt(self, query: str) -> str:
        """Builds the classification prompt, optionally reading from prompts.yaml.

        This avoids a hard dependency on PyYAML by doing a simple parse: if the
        file exists and contains a domain_classification block, use it; otherwise,
        return a default template.
        """
        default_template = (
            "You are a domain classifier. Determine if the following query is relevant to our knowledge base.\n\n"
            "Query: {query}\n\n"
            "Respond with either \"IN_DOMAIN\" or \"OUT_OF_DOMAIN\" only."
        )

        path = self.prompts_file_path
        if not os.path.exists(path):
            return default_template.format(query=query)

        try:
            # Lightweight extraction: find the domain_classification: | block
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            marker = "domain_classification: |"
            if marker not in content:
                return default_template.format(query=query)
            after = content.split(marker, 1)[1]
            # Collect indented lines until next top-level key
            lines = []
            for line in after.splitlines():
                if line.startswith(" ") or line.startswith("\t") or line.strip() == "":
                    lines.append(line.lstrip())
                else:
                    break
            template = "\n".join(lines).strip() or default_template
            return template.format(query=query)
        except Exception:
            return default_template.format(query=query)

    def _classify_with_openai(self, prompt: str, api_key: str) -> str:
        """Calls OpenAI Chat Completions API with a constrained instruction."""
        # Use a small, cost-effective model name; allow override via env
        model = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
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
                    "content": "Return exactly one token: IN_DOMAIN or OUT_OF_DOMAIN.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "max_tokens": 3,
        }
        resp = requests.post(url, headers=headers, json=data, timeout=20)
        resp.raise_for_status()
        out = resp.json()
        text = (
            out.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
            .upper()
        )
        # Normalize common variants
        if "IN_DOMAIN" in text:
            return "IN_DOMAIN"
        if "OUT_OF_DOMAIN" in text:
            return "OUT_OF_DOMAIN"
        # Defensive: try exact match fallback
        return "IN_DOMAIN" if text == "IN_DOMAIN" else "OUT_OF_DOMAIN"

    def _heuristic_classify(self, query: str) -> Literal["IN_DOMAIN", "OUT_OF_DOMAIN"]:
        """Simple keyword-based heuristic as a reliable fallback.

        Customize domain keywords to match your KB scope (e.g., product support).
        """
        query_lc = query.lower()
        in_keywords = (
            "account",
            "login",
            "password",
            "reset",
            "invoice",
            "subscription",
            "billing",
            "product",
            "feature",
            "api",
            "error",
            "bug",
            "support",
            "documentation",
        )
        out_keywords = (
            "weather",
            "news",
            "politics",
            "sports",
            "joke",
            "recipe",
            "song",
        )
        if any(k in query_lc for k in out_keywords):
            return "OUT_OF_DOMAIN"
        if any(k in query_lc for k in in_keywords):
            return "IN_DOMAIN"
        # Default conservatively to OUT_OF_DOMAIN to avoid bad retrieval
        return "OUT_OF_DOMAIN"