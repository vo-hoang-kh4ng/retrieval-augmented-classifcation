"""
Enhanced RAC Classifier with Class-aware Retrieval and Latent Fusion.
Combines discriminative retrieval with latent fusion for improved classification.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import json
import time
from dataclasses import dataclass

from ..retrieval.simple_vector_store import SimpleVectorStore
from ..retrieval.class_aware_retriever import ClassAwareRetriever, RetrievalResult
from ..fusion.latent_fusion import LatentFusionModule, FusionResult


@dataclass
class EnhancedClassificationResult:
    """Enhanced classification result with detailed explanations."""
    prediction: str
    confidence: float
    retrieval_results: List[RetrievalResult]
    fusion_result: FusionResult
    class_probabilities: Dict[str, float]
    explanation: Dict[str, Any]
    processing_time: float


class EnhancedRACClassifier:
    """
    Enhanced RAC Classifier that combines:
    1. Class-aware discriminative retrieval
    2. Latent fusion with cross-attention
    3. Advanced classification with long-tail support
    """
    
    def __init__(self,
                 embedding_dim: int = 128,
                 retrieval_k: int = 10,
                 fusion_strategy: str = "cross_attention",
                 alpha: float = 0.7,  # Retrieval: similarity vs discriminative
                 beta: float = 0.3,   # Retrieval: class frequency weight
                 gamma: float = 0.8,  # Classification: fusion vs similarity
                 use_hierarchical_fusion: bool = True,
                 long_tail_boost: float = 1.5):
        """
        Args:
            embedding_dim: Dimension of embeddings
            retrieval_k: Number of items to retrieve
            fusion_strategy: Fusion strategy ("cross_attention", "weighted_sum", "hierarchical")
            alpha: Balance between similarity and discriminative scores in retrieval
            beta: Weight for class frequency in retrieval
            gamma: Balance between fused embedding and direct similarity in classification
            use_hierarchical_fusion: Whether to use hierarchical fusion for multi-class scenarios
            long_tail_boost: Boost factor for long-tail classes
        """
        self.embedding_dim = embedding_dim
        self.retrieval_k = retrieval_k
        self.fusion_strategy = fusion_strategy
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.use_hierarchical_fusion = use_hierarchical_fusion
        self.long_tail_boost = long_tail_boost
        
        # Core components
        self.vector_store = SimpleVectorStore(embedding_dim=embedding_dim)
        self.class_aware_retriever = None
        self.fusion_module = None
        
        # Training data
        self.training_texts = []
        self.training_labels = []
        self.is_trained = False
        
        # Performance tracking
        self.performance_stats = {
            'total_predictions': 0,
            'avg_processing_time': 0.0,
            'class_performance': {},
            'long_tail_performance': {}
        }
    
    def fit(self, texts: List[str], labels: List[str]):
        """Train the enhanced classifier."""
        print("Training Enhanced RAC Classifier...")
        start_time = time.time()
        
        self.training_texts = texts
        self.training_labels = labels
        
        # Step 1: Build vector store
        print("  1. Building vector store...")
        self.vector_store.add_texts(texts, labels)
        
        # Step 2: Initialize and train class-aware retriever
        print("  2. Training class-aware retriever...")
        self.class_aware_retriever = ClassAwareRetriever(
            vector_store=self.vector_store,
            alpha=self.alpha,
            beta=self.beta,
            top_k=self.retrieval_k
        )
        self.class_aware_retriever.fit(texts, labels)
        
        # Step 3: Initialize fusion module
        print("  3. Initializing latent fusion module...")
        self.fusion_module = LatentFusionModule(
            embedding_dim=self.embedding_dim,
            fusion_strategy=self.fusion_strategy,
            use_position_encoding=True,
            use_score_weighting=True
        )
        
        # Step 4: Analyze class distribution for long-tail handling
        self._analyze_class_distribution()
        
        self.is_trained = True
        training_time = time.time() - start_time
        
        print(f"✓ Enhanced RAC Classifier trained in {training_time:.2f}s")
        print(f"  - {len(texts)} training examples")
        print(f"  - {len(set(labels))} unique classes")
        print(f"  - {len(self.long_tail_classes)} long-tail classes")
    
    def _analyze_class_distribution(self):
        """Analyze class distribution for long-tail handling."""
        class_counts = {}
        for label in self.training_labels:
            class_counts[label] = class_counts.get(label, 0) + 1
        
        total_samples = len(self.training_labels)
        long_tail_threshold = total_samples * 0.05  # Classes with < 5% of data
        
        self.class_counts = class_counts
        self.long_tail_classes = [
            cls for cls, count in class_counts.items() 
            if count < long_tail_threshold
        ]
        
        print(f"  - Long-tail classes ({len(self.long_tail_classes)}): {self.long_tail_classes[:5]}...")
    
    def _get_query_embedding(self, query: str) -> np.ndarray:
        """Get embedding for query text."""
        # Use the same embedding method as vector store
        return self.vector_store._text_to_embedding(query)
    
    def _compute_class_probabilities(self, 
                                   query_embedding: np.ndarray,
                                   fusion_result: FusionResult,
                                   retrieval_results: List[RetrievalResult]) -> Dict[str, float]:
        """Compute class probabilities using fused representation."""
        if not retrieval_results:
            return {}
        
        # Method 1: Direct similarity with fused embedding
        class_scores_fusion = {}
        
        # Get class centroids from retriever
        if hasattr(self.class_aware_retriever, 'class_centroids'):
            for class_label, centroid in self.class_aware_retriever.class_centroids.items():
                similarity = np.dot(fusion_result.fused_embedding, centroid) / (
                    np.linalg.norm(fusion_result.fused_embedding) * np.linalg.norm(centroid)
                )
                class_scores_fusion[class_label] = similarity
        
        # Method 2: Weighted voting from retrieved items
        class_scores_voting = {}
        for result in retrieval_results:
            if result.label not in class_scores_voting:
                class_scores_voting[result.label] = 0.0
            
            # Combine multiple scores
            vote_score = (
                0.4 * result.similarity_score +
                0.4 * result.discriminative_score +
                0.2 * result.combined_score
            )
            class_scores_voting[result.label] += vote_score
        
        # Normalize voting scores
        if class_scores_voting:
            max_vote = max(class_scores_voting.values())
            if max_vote > 0:
                class_scores_voting = {
                    k: v / max_vote for k, v in class_scores_voting.items()
                }
        
        # Combine both methods
        all_classes = set(class_scores_fusion.keys()) | set(class_scores_voting.keys())
        combined_scores = {}
        
        for class_label in all_classes:
            fusion_score = class_scores_fusion.get(class_label, 0.0)
            voting_score = class_scores_voting.get(class_label, 0.0)
            
            # Weighted combination
            combined_score = self.gamma * fusion_score + (1 - self.gamma) * voting_score
            
            # Apply long-tail boost
            if class_label in self.long_tail_classes:
                combined_score *= self.long_tail_boost
            
            combined_scores[class_label] = combined_score
        
        # Convert to probabilities (softmax)
        if combined_scores:
            scores_array = np.array(list(combined_scores.values()))
            exp_scores = np.exp(scores_array - np.max(scores_array))
            probabilities = exp_scores / np.sum(exp_scores)
            
            return {
                class_label: prob 
                for class_label, prob in zip(combined_scores.keys(), probabilities)
            }
        
        return {}
    
    def classify(self, query: str, explain: bool = True) -> EnhancedClassificationResult:
        """
        Classify query using enhanced RAC pipeline.
        
        Args:
            query: Query text to classify
            explain: Whether to include detailed explanations
            
        Returns:
            Enhanced classification result with explanations
        """
        if not self.is_trained:
            raise ValueError("Classifier must be trained before classification")
        
        start_time = time.time()
        
        # Step 1: Get query embedding
        query_embedding = self._get_query_embedding(query)
        
        # Step 2: Class-aware retrieval
        retrieval_results = self.class_aware_retriever.retrieve(query, k=self.retrieval_k)
        
        # Step 3: Latent fusion
        fusion_result = self.fusion_module.fuse(query_embedding, retrieval_results)
        
        # Step 4: Compute class probabilities
        class_probabilities = self._compute_class_probabilities(
            query_embedding, fusion_result, retrieval_results
        )
        
        # Step 5: Make prediction
        if class_probabilities:
            prediction = max(class_probabilities.items(), key=lambda x: x[1])[0]
            confidence = class_probabilities[prediction]
        else:
            prediction = "UNKNOWN"
            confidence = 0.0
        
        processing_time = time.time() - start_time
        
        # Step 6: Generate explanation
        explanation = {}
        if explain:
            explanation = self._generate_explanation(
                query, query_embedding, retrieval_results, 
                fusion_result, class_probabilities, processing_time
            )
        
        # Update performance stats
        self._update_performance_stats(prediction, processing_time)
        
        return EnhancedClassificationResult(
            prediction=prediction,
            confidence=confidence,
            retrieval_results=retrieval_results,
            fusion_result=fusion_result,
            class_probabilities=class_probabilities,
            explanation=explanation,
            processing_time=processing_time
        )
    
    def _generate_explanation(self,
                            query: str,
                            query_embedding: np.ndarray,
                            retrieval_results: List[RetrievalResult],
                            fusion_result: FusionResult,
                            class_probabilities: Dict[str, float],
                            processing_time: float) -> Dict[str, Any]:
        """Generate detailed explanation of the classification process."""
        
        # Retrieval explanation
        retrieval_explanation = self.class_aware_retriever.explain_retrieval(query, k=5)
        
        # Fusion explanation
        fusion_explanation = self.fusion_module.explain_fusion(fusion_result)
        
        # Classification explanation
        sorted_probs = sorted(class_probabilities.items(), key=lambda x: x[1], reverse=True)
        top_classes = sorted_probs[:5]
        
        explanation = {
            'query': query,
            'processing_time_ms': round(processing_time * 1000, 2),
            'pipeline_steps': {
                '1_retrieval': {
                    'strategy': 'class_aware_discriminative',
                    'num_retrieved': len(retrieval_results),
                    'candidate_classes': retrieval_explanation.get('candidate_classes', []),
                    'top_results': [
                        {
                            'text': r.text[:100] + '...' if len(r.text) > 100 else r.text,
                            'label': r.label,
                            'similarity': round(r.similarity_score, 3),
                            'discriminative': round(r.discriminative_score, 3),
                            'combined': round(r.combined_score, 3),
                            'is_long_tail': r.metadata.get('is_long_tail', False)
                        }
                        for r in retrieval_results[:3]
                    ]
                },
                '2_fusion': {
                    'strategy': fusion_result.fusion_metadata.get('strategy'),
                    'attention_weights': [round(w, 3) for w in fusion_result.attention_weights[:5]],
                    'attention_entropy': fusion_explanation['attention_analysis'].get('entropy', 0),
                    'fusion_metadata': fusion_result.fusion_metadata
                },
                '3_classification': {
                    'method': 'fused_embedding_with_voting',
                    'long_tail_boost_applied': any(
                        cls in self.long_tail_classes for cls, _ in top_classes
                    ),
                    'top_classes': [
                        {
                            'class': cls,
                            'probability': round(prob, 3),
                            'is_long_tail': cls in self.long_tail_classes,
                            'training_examples': self.class_counts.get(cls, 0)
                        }
                        for cls, prob in top_classes
                    ]
                }
            },
            'performance_insights': {
                'retrieval_quality': {
                    'avg_discriminative_score': round(
                        np.mean([r.discriminative_score for r in retrieval_results]), 3
                    ) if retrieval_results else 0,
                    'class_diversity': len(set(r.label for r in retrieval_results)),
                    'long_tail_coverage': sum(
                        1 for r in retrieval_results 
                        if r.label in self.long_tail_classes
                    )
                },
                'fusion_quality': {
                    'attention_concentration': fusion_explanation['attention_analysis'].get('max_weight', 0),
                    'information_diversity': fusion_explanation['attention_analysis'].get('entropy', 0)
                }
            }
        }
        
        return explanation
    
    def _update_performance_stats(self, prediction: str, processing_time: float):
        """Update performance statistics."""
        self.performance_stats['total_predictions'] += 1
        
        # Update average processing time
        total = self.performance_stats['total_predictions']
        current_avg = self.performance_stats['avg_processing_time']
        self.performance_stats['avg_processing_time'] = (
            (current_avg * (total - 1) + processing_time) / total
        )
        
        # Update class-specific stats
        if prediction not in self.performance_stats['class_performance']:
            self.performance_stats['class_performance'][prediction] = {
                'count': 0,
                'avg_time': 0.0
            }
        
        class_stats = self.performance_stats['class_performance'][prediction]
        class_stats['count'] += 1
        class_stats['avg_time'] = (
            (class_stats['avg_time'] * (class_stats['count'] - 1) + processing_time) / 
            class_stats['count']
        )
        
        # Update long-tail stats
        if prediction in self.long_tail_classes:
            if 'long_tail_predictions' not in self.performance_stats['long_tail_performance']:
                self.performance_stats['long_tail_performance']['long_tail_predictions'] = 0
            self.performance_stats['long_tail_performance']['long_tail_predictions'] += 1
    
    def batch_classify(self, queries: List[str]) -> List[EnhancedClassificationResult]:
        """Classify multiple queries efficiently."""
        results = []
        for query in queries:
            result = self.classify(query, explain=False)  # Skip explanations for speed
            results.append(result)
        return results
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary and statistics."""
        stats = self.performance_stats.copy()
        
        # Add class distribution analysis
        if self.class_counts:
            stats['class_distribution'] = {
                'total_classes': len(self.class_counts),
                'long_tail_classes': len(self.long_tail_classes),
                'class_balance': {
                    'most_frequent': max(self.class_counts.items(), key=lambda x: x[1]),
                    'least_frequent': min(self.class_counts.items(), key=lambda x: x[1]),
                    'median_frequency': np.median(list(self.class_counts.values()))
                }
            }
        
        # Add retriever statistics
        if self.class_aware_retriever:
            stats['retriever_stats'] = self.class_aware_retriever.get_class_statistics()
        
        return stats
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        model_data = {
            'config': {
                'embedding_dim': self.embedding_dim,
                'retrieval_k': self.retrieval_k,
                'fusion_strategy': self.fusion_strategy,
                'alpha': self.alpha,
                'beta': self.beta,
                'gamma': self.gamma,
                'long_tail_boost': self.long_tail_boost
            },
            'training_data': {
                'texts': self.training_texts,
                'labels': self.training_labels
            },
            'class_stats': {
                'class_counts': self.class_counts,
                'long_tail_classes': self.long_tail_classes
            },
            'performance_stats': self.performance_stats
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(model_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model."""
        with open(filepath, 'r', encoding='utf-8') as f:
            model_data = json.load(f)
        
        # Restore config
        config = model_data['config']
        self.embedding_dim = config['embedding_dim']
        self.retrieval_k = config['retrieval_k']
        self.fusion_strategy = config['fusion_strategy']
        self.alpha = config['alpha']
        self.beta = config['beta']
        self.gamma = config['gamma']
        self.long_tail_boost = config['long_tail_boost']
        
        # Restore training data and retrain
        training_data = model_data['training_data']
        self.fit(training_data['texts'], training_data['labels'])
        
        # Restore additional stats
        self.class_counts = model_data['class_stats']['class_counts']
        self.long_tail_classes = model_data['class_stats']['long_tail_classes']
        self.performance_stats = model_data['performance_stats']
        
        print(f"✓ Model loaded from {filepath}")
