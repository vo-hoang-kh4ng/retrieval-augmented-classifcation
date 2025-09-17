from typing import Optional
from dataclasses import dataclass
import os
import json

@dataclass
class LLMClients:
    openai_api_key: Optional[str]
    google_api_key: Optional[str]

    @classmethod
    def from_env(cls) -> "LLMClients":
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )

@dataclass
class Settings:
    """Configuration settings loaded from hyperparameters.json"""
    retrieval_top_k: int = 5
    similarity_threshold: float = 0.7
    knn_weights: str = "uniform"
    embedding_model: str = "text-embedding-ada-002"
    embedding_dimension: int = 1536
    embedding_batch_size: int = 100
    llm_model: str = "gpt-3.5-turbo"
    temperature: float = 0.1
    max_tokens: int = 1000
    naics_shortlist_k: int = 10
    confidence_threshold: float = 0.8
    log_level: str = "INFO"
    log_file: str = "./logs/rag_rac_naics.log"

    @classmethod
    def from_file(cls, config_path: str = "config/hyperparameters.json") -> "Settings":
        """Load settings from configuration file"""
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                return cls(
                    retrieval_top_k=config.get("retrieval", {}).get("top_k", 5),
                    similarity_threshold=config.get("retrieval", {}).get("similarity_threshold", 0.7),
                    knn_weights=config.get("retrieval", {}).get("knn_weights", "uniform"),
                    embedding_model=config.get("embeddings", {}).get("model", "text-embedding-ada-002"),
                    embedding_dimension=config.get("embeddings", {}).get("dimension", 1536),
                    embedding_batch_size=config.get("embeddings", {}).get("batch_size", 100),
                    llm_model=config.get("llm", {}).get("model", "gpt-3.5-turbo"),
                    temperature=config.get("llm", {}).get("temperature", 0.1),
                    max_tokens=config.get("llm", {}).get("max_tokens", 1000),
                    naics_shortlist_k=config.get("naics", {}).get("shortlist_k", 10),
                    confidence_threshold=config.get("naics", {}).get("confidence_threshold", 0.8),
                    log_level=config.get("logging", {}).get("level", "INFO"),
                    log_file=config.get("logging", {}).get("file", "./logs/rag_rac_naics.log")
                )
            except Exception as e:
                print(f"Warning: Could not load config from {config_path}: {e}")
                return cls()
        return cls()
