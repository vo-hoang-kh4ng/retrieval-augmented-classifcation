# KNN Classifier for retrieval-based classification
import numpy as np
from typing import List, Tuple, Optional
from sklearn.neighbors import KNeighborsClassifier

class KNNClassifier:
    def __init__(self, k: int = 5, weights: str = "uniform"):
        self.k = k
        self.weights = weights
        self.knn = KNeighborsClassifier(n_neighbors=k, weights=weights)
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: List[str]) -> None:
        """Fit KNN on embeddings and labels"""
        self.knn.fit(X, y)
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> List[str]:
        """Predict labels using mode of k-nearest neighbors"""
        if not self.is_fitted:
            raise ValueError("Must fit before predict")
        return self.knn.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities"""
        if not self.is_fitted:
            raise ValueError("Must fit before predict_proba")
        return self.knn.predict_proba(X)
    
    def get_neighbors(self, X: np.ndarray, k: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Get k nearest neighbors and distances"""
        if k is None:
            k = self.k
        distances, indices = self.knn.kneighbors(X, n_neighbors=k)
        return distances, indices
