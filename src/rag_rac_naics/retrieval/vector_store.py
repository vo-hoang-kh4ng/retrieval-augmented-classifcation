"""Vector Store Management (ChromaDB wrapper)."""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import chromadb
from chromadb.config import Settings

class VectorStore:
    def __init__(self, db_path: str = "./data/processed/chroma_db", collection_name: str = "default"):
        self.db_path = db_path
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(name=collection_name)
    
    def add_texts(self, texts: List[str], embeddings: np.ndarray, labels: Optional[List[str]] = None) -> None:
        """Add texts with embeddings to vector store"""
        existing = self.collection.count()
        ids = [f"doc_{existing + i}" for i in range(len(texts))]
        metadatas = [{"label": label} for label in (labels or [""] * len(texts))]
        
        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas
        )
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> Tuple[List[str], List[str], List[float]]:
        """Search for similar texts"""
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k
        )
        
        texts = results["documents"][0]
        labels = [metadata.get("label", "") for metadata in results["metadatas"][0]]
        scores = [1.0 - distance for distance in results["distances"][0]]
        
        return texts, labels, scores
    
    def get_all_texts(self) -> Tuple[List[str], np.ndarray, List[str]]:
        """Get all texts, embeddings, and labels"""
        results = self.collection.get()
        texts = results["documents"]
        labels = [metadata.get("label", "") for metadata in results["metadatas"]]
        embeddings = np.array(results["embeddings"])
        
        return texts, embeddings, labels
    
    def clear(self) -> None:
        """Clear all data from collection"""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(name=self.collection_name)
