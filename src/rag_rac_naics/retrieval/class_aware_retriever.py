"""
Class-aware Retriever with discriminative ranking.
Prioritizes evidences that help distinguish between candidate NAICS classes.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
import json
import math
from dataclasses import dataclass

from .simple_vector_store import SimpleVectorStore


@dataclass
class RetrievalResult:
    """Enhanced retrieval result with discriminative scores."""
    text: str
    label: str
    similarity_score: float
    discriminative_score: float
    combined_score: float
    class_frequency: int
    metadata: Dict[str, Any] = None


class ClassAwareRetriever:
    """
    Retriever that learns to prioritize discriminative evidences.
    
    Key innovations:
    1. Class frequency weighting for long-tail support
    2. Discriminative scoring based on class separation
    3. Adaptive reranking based on candidate classes
    """
    
    def __init__(self, 
                 vector_store: SimpleVectorStore,
                 alpha: float = 0.7,  # Weight for similarity vs discriminative score
                 beta: float = 0.3,   # Weight for class frequency
                 top_k: int = 10):
        """
        Args:
            vector_store: Base vector store for similarity search
            alpha: Balance between similarity and discriminative scores
            beta: Weight for class frequency (long-tail support)
            top_k: Number of items to retrieve
        """
        self.vector_store = vector_store
        self.alpha = alpha
        self.beta = beta
        self.top_k = top_k
        
        # Class statistics for discriminative scoring
        self.class_stats = {}
        self.class_centroids = {}
        self.inter_class_distances = {}
        
        # Training data for discriminative learning
        self.training_queries = []
        self.training_labels = []
        
    def fit(self, queries: List[str], labels: List[str]):
        """Train the discriminative components."""
        self.training_queries = queries
        self.training_labels = labels
        
        # Compute class statistics
        self._compute_class_statistics()
        
        # Compute class centroids in embedding space
        self._compute_class_centroids()
        
        # Compute inter-class distances for discriminative scoring
        self._compute_inter_class_distances()
        
        print(f"✓ Class-aware retriever trained on {len(queries)} queries")
        print(f"✓ Found {len(self.class_stats)} classes")
        
    def _compute_class_statistics(self):
        """Compute frequency and other stats for each class."""
        class_counts = defaultdict(int)
        
        for label in self.training_labels:
            class_counts[label] += 1
            
        total_samples = len(self.training_labels)
        
        for class_label, count in class_counts.items():
            self.class_stats[class_label] = {
                'frequency': count,
                'relative_frequency': count / total_samples,
                'inverse_frequency': math.log(total_samples / count),  # IDF-like
                'is_long_tail': count < (total_samples * 0.05)  # < 5% of data
            }
    
    def _compute_class_centroids(self):
        """Compute centroid embeddings for each class."""
        if not self.vector_store.texts:
            return
            
        class_embeddings = defaultdict(list)
        
        # Group embeddings by class
        for i, (text, label) in enumerate(zip(self.vector_store.texts, self.vector_store.labels)):
            if i < len(self.vector_store.embeddings):
                class_embeddings[label].append(self.vector_store.embeddings[i])
        
        # Compute centroids
        for class_label, embeddings in class_embeddings.items():
            if embeddings:
                centroid = np.mean(embeddings, axis=0)
                self.class_centroids[class_label] = centroid
    
    def _compute_inter_class_distances(self):
        """Compute distances between class centroids for discriminative scoring."""
        classes = list(self.class_centroids.keys())
        
        for i, class1 in enumerate(classes):
            for j, class2 in enumerate(classes):
                if i != j:
                    dist = self._cosine_distance(
                        self.class_centroids[class1], 
                        self.class_centroids[class2]
                    )
                    self.inter_class_distances[(class1, class2)] = dist
    
    def _cosine_distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine distance between two vectors."""
        similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        return 1 - similarity
    
    def _get_candidate_classes(self, query: str, top_classes: int = 5) -> List[str]:
        """Get candidate classes for the query based on initial similarity."""
        # Get initial retrieval results
        initial_results = self.vector_store.search(query, k=self.top_k * 2)
        
        # Count class frequencies in top results
        class_scores = defaultdict(float)
        for result in initial_results:
            class_scores[result['label']] += result['score']
        
        # Sort by aggregated scores
        sorted_classes = sorted(class_scores.items(), key=lambda x: x[1], reverse=True)
        
        return [class_label for class_label, _ in sorted_classes[:top_classes]]
    
    def _compute_discriminative_score(self, 
                                    item_embedding: np.ndarray,
                                    item_label: str,
                                    candidate_classes: List[str]) -> float:
        """
        Compute discriminative score for an item.
        Higher score = better at distinguishing between candidate classes.
        """
        if item_label not in self.class_centroids:
            return 0.0
        
        # Distance to own class centroid (should be small)
        own_class_distance = self._cosine_distance(
            item_embedding, 
            self.class_centroids[item_label]
        )
        
        # Average distance to other candidate class centroids (should be large)
        other_class_distances = []
        for candidate_class in candidate_classes:
            if candidate_class != item_label and candidate_class in self.class_centroids:
                dist = self._cosine_distance(
                    item_embedding,
                    self.class_centroids[candidate_class]
                )
                other_class_distances.append(dist)
        
        if not other_class_distances:
            return 0.0
        
        avg_other_distance = np.mean(other_class_distances)
        
        # Discriminative score: high when close to own class, far from others
        discriminative_score = avg_other_distance - own_class_distance
        
        return max(0.0, discriminative_score)  # Ensure non-negative
    
    def _compute_class_frequency_weight(self, class_label: str) -> float:
        """Compute weight to boost long-tail classes."""
        if class_label not in self.class_stats:
            return 1.0
        
        stats = self.class_stats[class_label]
        
        # Boost long-tail classes
        if stats['is_long_tail']:
            return 1.0 + stats['inverse_frequency'] * 0.1
        else:
            return 1.0
    
    def retrieve(self, query: str, k: Optional[int] = None) -> List[RetrievalResult]:
        """
        Retrieve with class-aware discriminative ranking.
        
        Args:
            query: Query text
            k: Number of results to return (default: self.top_k)
            
        Returns:
            List of RetrievalResult with discriminative scores
        """
        k = k or self.top_k
        
        # Step 1: Get candidate classes
        candidate_classes = self._get_candidate_classes(query)
        
        # Step 2: Get initial similarity-based results (more than needed)
        initial_results = self.vector_store.search(query, k=k*3)
        
        # Step 3: Rerank with discriminative scoring
        enhanced_results = []
        
        for result in initial_results:
            # Get item embedding
            item_idx = None
            for i, text in enumerate(self.vector_store.texts):
                if text == result['text']:
                    item_idx = i
                    break
            
            if item_idx is None or item_idx >= len(self.vector_store.embeddings):
                continue
                
            item_embedding = np.array(self.vector_store.embeddings[item_idx])
            
            # Compute discriminative score
            discriminative_score = self._compute_discriminative_score(
                item_embedding, 
                result['label'], 
                candidate_classes
            )
            
            # Compute class frequency weight
            freq_weight = self._compute_class_frequency_weight(result['label'])
            
            # Combine scores
            similarity_score = result['score']
            combined_score = (
                self.alpha * similarity_score + 
                (1 - self.alpha) * discriminative_score +
                self.beta * freq_weight
            )
            
            enhanced_result = RetrievalResult(
                text=result['text'],
                label=result['label'],
                similarity_score=similarity_score,
                discriminative_score=discriminative_score,
                combined_score=combined_score,
                class_frequency=self.class_stats.get(result['label'], {}).get('frequency', 0),
                metadata={
                    'candidate_classes': candidate_classes,
                    'freq_weight': freq_weight,
                    'is_long_tail': self.class_stats.get(result['label'], {}).get('is_long_tail', False)
                }
            )
            
            enhanced_results.append(enhanced_result)
        
        # Step 4: Sort by combined score and return top k
        enhanced_results.sort(key=lambda x: x.combined_score, reverse=True)
        
        return enhanced_results[:k]
    
    def get_class_statistics(self) -> Dict[str, Any]:
        """Get class statistics for analysis."""
        return {
            'class_stats': self.class_stats,
            'num_classes': len(self.class_stats),
            'long_tail_classes': [
                cls for cls, stats in self.class_stats.items() 
                if stats.get('is_long_tail', False)
            ],
            'inter_class_distances': dict(self.inter_class_distances)
        }
    
    def explain_retrieval(self, query: str, k: int = 5) -> Dict[str, Any]:
        """Explain why certain items were retrieved."""
        results = self.retrieve(query, k)
        candidate_classes = self._get_candidate_classes(query)
        
        explanation = {
            'query': query,
            'candidate_classes': candidate_classes,
            'retrieval_results': []
        }
        
        for result in results:
            result_explanation = {
                'text': result.text[:100] + '...' if len(result.text) > 100 else result.text,
                'label': result.label,
                'scores': {
                    'similarity': round(result.similarity_score, 3),
                    'discriminative': round(result.discriminative_score, 3),
                    'combined': round(result.combined_score, 3)
                },
                'class_info': {
                    'frequency': result.class_frequency,
                    'is_long_tail': result.metadata.get('is_long_tail', False),
                    'freq_weight': round(result.metadata.get('freq_weight', 1.0), 3)
                }
            }
            explanation['retrieval_results'].append(result_explanation)
        
        return explanation
