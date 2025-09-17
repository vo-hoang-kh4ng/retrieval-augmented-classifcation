"""Simple in-memory vector store as fallback when ChromaDB is not available."""

import numpy as np
import json
import os
from typing import List, Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class SimpleVectorStore:
    """In-memory vector store using numpy for similarity search."""
    
    def __init__(self, db_path: str = "./data/processed/simple_db", collection_name: str = "default"):
        self.db_path = db_path
        self.collection_name = collection_name
        self.texts: List[str] = []
        self.embeddings: np.ndarray = np.array([]).reshape(0, 0)
        self.metadatas: List[Dict[str, Any]] = []
        self.ids: List[str] = []
        
        # Try to load existing data
        self._load_from_disk()
        
    def _get_file_path(self) -> str:
        """Get the file path for persisting data."""
        os.makedirs(self.db_path, exist_ok=True)
        return os.path.join(self.db_path, f"{self.collection_name}.json")
    
    def _load_from_disk(self):
        """Load data from disk if it exists."""
        file_path = self._get_file_path()
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                self.texts = data.get('texts', [])
                self.metadatas = data.get('metadatas', [])
                self.ids = data.get('ids', [])
                
                # Load embeddings
                embeddings_list = data.get('embeddings', [])
                if embeddings_list:
                    self.embeddings = np.array(embeddings_list, dtype=float)
                else:
                    self.embeddings = np.array([]).reshape(0, 0)
                    
                logger.info(f"Loaded {len(self.texts)} documents from {file_path}")
            except Exception as e:
                logger.warning(f"Failed to load from {file_path}: {e}")
                self._reset()
    
    def _save_to_disk(self):
        """Save data to disk."""
        file_path = self._get_file_path()
        try:
            data = {
                'texts': self.texts,
                'metadatas': self.metadatas,
                'ids': self.ids,
                'embeddings': self.embeddings.tolist() if self.embeddings.size > 0 else []
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
            logger.info(f"Saved {len(self.texts)} documents to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save to {file_path}: {e}")
    
    def _reset(self):
        """Reset all data."""
        self.texts = []
        self.embeddings = np.array([]).reshape(0, 0)
        self.metadatas = []
        self.ids = []
    
    def add_texts(self, texts: List[str], embeddings: np.ndarray, labels: List[str], ids: Optional[List[str]] = None) -> None:
        """Add texts with embeddings and labels."""
        metadatas = [{"label": label} for label in labels]
        self.upsert_texts(texts, embeddings, metadatas, ids)
    
    def upsert_texts(self, texts: List[str], embeddings: np.ndarray, metadatas: Optional[List[Dict[str, Any]]] = None, ids: Optional[List[str]] = None) -> None:
        """Add or update texts with embeddings."""
        if len(texts) != embeddings.shape[0]:
            raise ValueError("Number of texts must match number of embeddings")
        
        if ids is None:
            ids = [f"doc_{len(self.texts) + i}" for i in range(len(texts))]
        
        if metadatas is None:
            metadatas = [{}] * len(texts)
        
        # Initialize embeddings array if empty
        if self.embeddings.size == 0:
            self.embeddings = embeddings.copy()
            self.texts = texts.copy()
            self.metadatas = metadatas.copy()
            self.ids = ids.copy()
        else:
            # Check dimension compatibility
            if embeddings.shape[1] != self.embeddings.shape[1]:
                raise ValueError(f"Embedding dimension mismatch: expected {self.embeddings.shape[1]}, got {embeddings.shape[1]}")
            
            # Append new data
            self.embeddings = np.vstack([self.embeddings, embeddings])
            self.texts.extend(texts)
            self.metadatas.extend(metadatas)
            self.ids.extend(ids)
        
        # Save to disk
        self._save_to_disk()
        logger.info(f"Added {len(texts)} documents to collection '{self.collection_name}'")
    
    def search(self, query_embedding: np.ndarray, k: int = 5, where_filter: Optional[Dict[str, Any]] = None) -> Tuple[List[str], List[Dict[str, Any]], List[float]]:
        """Search for similar documents."""
        if self.embeddings.size == 0:
            return [], [], []
        
        # Ensure query_embedding is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Calculate cosine similarity
        # Normalize embeddings
        query_norm = query_embedding / (np.linalg.norm(query_embedding, axis=1, keepdims=True) + 1e-8)
        doc_norms = self.embeddings / (np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-8)
        
        # Compute similarities
        similarities = np.dot(doc_norms, query_norm.T).flatten()
        
        # Apply filters if specified
        valid_indices = list(range(len(self.texts)))
        if where_filter:
            valid_indices = []
            for i, metadata in enumerate(self.metadatas):
                match = True
                for key, value in where_filter.items():
                    if metadata.get(key) != value:
                        match = False
                        break
                if match:
                    valid_indices.append(i)
        
        if not valid_indices:
            return [], [], []
        
        # Get similarities for valid indices
        valid_similarities = [(i, similarities[i]) for i in valid_indices]
        
        # Sort by similarity (descending)
        valid_similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Get top k
        top_k = valid_similarities[:k]
        
        # Extract results
        result_texts = [self.texts[i] for i, _ in top_k]
        result_metadatas = [self.metadatas[i] for i, _ in top_k]
        result_scores = [float(score) for _, score in top_k]
        
        return result_texts, result_metadatas, result_scores
    
    def get_collection_count(self) -> int:
        """Get the number of documents in the collection."""
        return len(self.texts)
    
    def clear_collection(self) -> None:
        """Clear all documents from the collection."""
        self._reset()
        self._save_to_disk()
        logger.info(f"Cleared collection '{self.collection_name}'")

# Alias for compatibility
VectorStore = SimpleVectorStore
