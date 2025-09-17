"""Vector Store Management (ChromaDB wrapper)."""

import numpy as np
import uuid
import logging
from typing import List, Tuple, Optional, Dict, Any
import chromadb

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class VectorStore:
    """A wrapper class for ChromaDB to manage vector storage and retrieval."""

    def __init__(self, db_path: str = "./data/processed/chroma_db", collection_name: str = "default"):
        """
        Initializes the VectorStore.

        Args:
            db_path (str): Path to the persistent database directory.
            collection_name (str): Name of the collection to use.
        """
        self.db_path = db_path
        self.collection_name = collection_name
        try:
            # Sử dụng PersistentClient để lưu trữ dữ liệu trên đĩa
            self.client = chromadb.PersistentClient(path=db_path)
            # Lấy hoặc tạo collection. Có thể thêm metadata để chỉ định distance metric
            # ví dụ: metadata={"hnsw:space": "cosine"}
            self.collection = self.client.get_or_create_collection(name=collection_name)
            logging.info(f"VectorStore initialized successfully. Collection '{collection_name}' with {self.collection.count()} documents.")
        except Exception as e:
            logging.error(f"Failed to initialize ChromaDB client at path {db_path}: {e}")
            raise

    def add_texts(self, texts: List[str], embeddings: np.ndarray, labels: List[str], ids: Optional[List[str]] = None) -> None:
        """
        Add texts with embeddings and labels to the vector store.
        This is a convenience method that formats labels as metadata.

        Args:
            texts (List[str]): The list of texts to add.
            embeddings (np.ndarray): The corresponding embeddings for the texts.
            labels (List[str]): The corresponding labels for the texts.
            ids (Optional[List[str]]): A list of unique IDs for the texts. If None, UUIDs are generated.
        """
        metadatas = [{"label": label} for label in labels]
        self.upsert_texts(texts, embeddings, metadatas, ids)

    def upsert_texts(self, texts: List[str], embeddings: np.ndarray, metadatas: Optional[List[Dict[str, Any]]] = None, ids: Optional[List[str]] = None) -> None:
        """
        Add or update texts with embeddings to the vector store.
        Using upsert is safer than add as it handles existing IDs.

        Args:
            texts (List[str]): The list of texts to add.
            embeddings (np.ndarray): The corresponding embeddings for the texts.
            metadatas (Optional[List[Dict]]): A list of metadata dictionaries.
            ids (Optional[List[str]]): A list of unique IDs for the texts. If None, UUIDs are generated.
        """
        if ids is None:
            # Cải tiến: Sử dụng UUID để đảm bảo ID là duy nhất và ổn định
            ids = [str(uuid.uuid4()) for _ in texts]

        if len(texts) != len(ids) or len(texts) != embeddings.shape[0]:
            raise ValueError("The number of texts, embeddings, and IDs must be the same.")

        # Cải tiến: Cho phép metadata linh hoạt hơn, không chỉ là 'label'
        if metadatas is None:
            metadatas = [{}] * len(texts)

        try:
            self.collection.upsert(
                ids=ids,
                documents=texts,
                embeddings=embeddings.tolist(),
                metadatas=metadatas
            )
            logging.info(f"Upserted {len(texts)} documents into '{self.collection_name}'.")
        except Exception as e:
            logging.error(f"Failed to upsert documents: {e}")
            raise

    def search(self, query_embedding: np.ndarray, k: int = 5, where_filter: Optional[Dict[str, Any]] = None) -> Tuple[List[str], List[Dict[str, Any]], List[float]]:
        """
        Search for the k most similar texts to a given query embedding.

        Args:
            query_embedding (np.ndarray): The embedding of the query text.
            k (int): The number of similar texts to retrieve.
            where_filter (Optional[Dict]): A filter to apply on the metadata.

        Returns:
            Tuple containing lists of documents, metadatas, and similarity scores.
        """
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=k,
                where=where_filter
            )

            if not results["ids"][0]:
                return [], [], []

            documents = results["documents"][0]
            metadatas = results["metadatas"][0]
            # ChromaDB distances are often squared L2. For similarity, 1 - distance is a common conversion.
            # If using cosine, distance = 1 - similarity, so similarity = 1 - distance.
            scores = [1.0 - dist for dist in results["distances"][0]]

            return documents, metadatas, scores
        except Exception as e:
            logging.error(f"Failed to perform search: {e}")
            return [], [], []

    def get_collection_count(self) -> int:
        """Returns the total number of items in the collection."""
        return self.collection.count()

    def clear_collection(self) -> None:
        """
        Clears all data from the collection, effectively resetting it.
        """
        logging.warning(f"Clearing all data from collection '{self.collection_name}'.")
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.get_or_create_collection(name=self.collection_name)
            logging.info(f"Collection '{self.collection_name}' has been cleared.")
        except Exception as e:
            logging.error(f"Failed to clear collection '{self.collection_name}': {e}")
            raise
