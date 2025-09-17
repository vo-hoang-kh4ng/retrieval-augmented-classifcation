"""
Latent Fusion Module with cross-attention.
Fuses retrieved information in latent space instead of raw text concatenation.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import json
import math
from dataclasses import dataclass

from ..retrieval.class_aware_retriever import RetrievalResult


@dataclass
class FusionResult:
    """Result from latent fusion."""
    fused_embedding: np.ndarray
    attention_weights: List[float]
    retrieved_items: List[RetrievalResult]
    fusion_metadata: Dict[str, Any]


class MultiHeadAttention:
    """Simple multi-head attention implementation."""
    
    def __init__(self, d_model: int, num_heads: int = 4):
        """
        Args:
            d_model: Dimension of embeddings
            num_heads: Number of attention heads
        """
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Initialize weight matrices (simplified - in practice use proper initialization)
        self.W_q = np.random.normal(0, 0.1, (d_model, d_model))
        self.W_k = np.random.normal(0, 0.1, (d_model, d_model))
        self.W_v = np.random.normal(0, 0.1, (d_model, d_model))
        self.W_o = np.random.normal(0, 0.1, (d_model, d_model))
        
    def forward(self, 
                query: np.ndarray, 
                keys: np.ndarray, 
                values: np.ndarray,
                mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass of multi-head attention.
        
        Args:
            query: Query vector [d_model]
            keys: Key vectors [seq_len, d_model]
            values: Value vectors [seq_len, d_model]
            mask: Optional mask [seq_len]
            
        Returns:
            output: Attended output [d_model]
            attention_weights: Attention weights [seq_len]
        """
        seq_len = keys.shape[0]
        
        # Linear transformations
        Q = np.dot(query, self.W_q).reshape(1, self.num_heads, self.d_k)
        K = np.dot(keys, self.W_k).reshape(seq_len, self.num_heads, self.d_k)
        V = np.dot(values, self.W_v).reshape(seq_len, self.num_heads, self.d_k)
        
        # Scaled dot-product attention for each head
        head_outputs = []
        all_attention_weights = []
        
        for h in range(self.num_heads):
            q_h = Q[0, h, :]  # [d_k]
            k_h = K[:, h, :]  # [seq_len, d_k]
            v_h = V[:, h, :]  # [seq_len, d_k]
            
            # Attention scores
            scores = np.dot(k_h, q_h) / math.sqrt(self.d_k)  # [seq_len]
            
            # Apply mask if provided
            if mask is not None:
                scores = scores + mask * -1e9
            
            # Softmax
            attention_weights = self._softmax(scores)
            all_attention_weights.append(attention_weights)
            
            # Weighted sum
            head_output = np.sum(v_h * attention_weights.reshape(-1, 1), axis=0)
            head_outputs.append(head_output)
        
        # Concatenate heads
        concat_output = np.concatenate(head_outputs)
        
        # Final linear transformation
        output = np.dot(concat_output, self.W_o)
        
        # Average attention weights across heads
        avg_attention_weights = np.mean(all_attention_weights, axis=0)
        
        return output, avg_attention_weights
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)


class LatentFusionModule:
    """
    Latent Fusion Module that combines query and retrieved items using attention.
    
    Key features:
    1. Cross-attention between query and retrieved items
    2. Position-aware fusion with retrieval scores
    3. Adaptive weighting based on discriminative scores
    4. Hierarchical fusion for different granularities
    """
    
    def __init__(self, 
                 embedding_dim: int = 128,
                 num_heads: int = 4,
                 fusion_strategy: str = "cross_attention",
                 use_position_encoding: bool = True,
                 use_score_weighting: bool = True):
        """
        Args:
            embedding_dim: Dimension of embeddings
            num_heads: Number of attention heads
            fusion_strategy: Strategy for fusion ("cross_attention", "weighted_sum", "hierarchical")
            use_position_encoding: Whether to use position encoding for retrieved items
            use_score_weighting: Whether to weight by retrieval scores
        """
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.fusion_strategy = fusion_strategy
        self.use_position_encoding = use_position_encoding
        self.use_score_weighting = use_score_weighting
        
        # Initialize attention module
        self.attention = MultiHeadAttention(embedding_dim, num_heads)
        
        # Learnable parameters for fusion
        self.query_projection = np.random.normal(0, 0.1, (embedding_dim, embedding_dim))
        self.item_projection = np.random.normal(0, 0.1, (embedding_dim, embedding_dim))
        self.fusion_gate = np.random.normal(0, 0.1, (embedding_dim * 2, embedding_dim))
        
        # Position encoding
        if use_position_encoding:
            self.position_encoding = self._create_position_encoding(max_len=50)
    
    def _create_position_encoding(self, max_len: int) -> np.ndarray:
        """Create sinusoidal position encoding."""
        pe = np.zeros((max_len, self.embedding_dim))
        
        position = np.arange(0, max_len).reshape(-1, 1)
        div_term = np.exp(np.arange(0, self.embedding_dim, 2) * 
                         -(math.log(10000.0) / self.embedding_dim))
        
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        return pe
    
    def _add_position_encoding(self, embeddings: np.ndarray) -> np.ndarray:
        """Add position encoding to embeddings."""
        seq_len = embeddings.shape[0]
        if self.use_position_encoding and seq_len <= self.position_encoding.shape[0]:
            return embeddings + self.position_encoding[:seq_len]
        return embeddings
    
    def _create_score_mask(self, retrieval_results: List[RetrievalResult]) -> np.ndarray:
        """Create attention mask based on retrieval scores."""
        if not self.use_score_weighting:
            return np.zeros(len(retrieval_results))
        
        # Convert scores to mask (higher scores = less masking)
        scores = np.array([result.combined_score for result in retrieval_results])
        
        # Normalize scores to [0, 1]
        if scores.max() > scores.min():
            normalized_scores = (scores - scores.min()) / (scores.max() - scores.min())
        else:
            normalized_scores = np.ones_like(scores)
        
        # Convert to mask (0 = no masking, negative = masking)
        mask = (normalized_scores - 1) * 10  # Scale for stronger effect
        
        return mask
    
    def fuse_cross_attention(self, 
                           query_embedding: np.ndarray,
                           retrieval_results: List[RetrievalResult]) -> FusionResult:
        """Fuse using cross-attention mechanism."""
        if not retrieval_results:
            return FusionResult(
                fused_embedding=query_embedding,
                attention_weights=[],
                retrieved_items=[],
                fusion_metadata={'strategy': 'no_items'}
            )
        
        # Prepare retrieved item embeddings
        item_embeddings = []
        for result in retrieval_results:
            # In practice, you'd get actual embeddings from the vector store
            # For now, create dummy embeddings based on text hash
            dummy_embedding = self._text_to_embedding(result.text)
            item_embeddings.append(dummy_embedding)
        
        item_embeddings = np.array(item_embeddings)
        
        # Add position encoding
        item_embeddings = self._add_position_encoding(item_embeddings)
        
        # Project embeddings
        query_proj = np.dot(query_embedding, self.query_projection)
        items_proj = np.dot(item_embeddings, self.item_projection)
        
        # Create score-based mask
        score_mask = self._create_score_mask(retrieval_results)
        
        # Cross-attention: query attends to retrieved items
        attended_output, attention_weights = self.attention.forward(
            query=query_proj,
            keys=items_proj,
            values=items_proj,
            mask=score_mask
        )
        
        # Gated fusion of query and attended items
        fusion_input = np.concatenate([query_proj, attended_output])
        gate_weights = self._sigmoid(np.dot(fusion_input, self.fusion_gate))
        
        # Final fused embedding
        fused_embedding = gate_weights * query_proj + (1 - gate_weights) * attended_output
        
        return FusionResult(
            fused_embedding=fused_embedding,
            attention_weights=attention_weights.tolist(),
            retrieved_items=retrieval_results,
            fusion_metadata={
                'strategy': 'cross_attention',
                'num_items': len(retrieval_results),
                'avg_attention_entropy': self._compute_entropy(attention_weights),
                'gate_weights_mean': np.mean(gate_weights)
            }
        )
    
    def fuse_weighted_sum(self, 
                         query_embedding: np.ndarray,
                         retrieval_results: List[RetrievalResult]) -> FusionResult:
        """Fuse using weighted sum based on discriminative scores."""
        if not retrieval_results:
            return FusionResult(
                fused_embedding=query_embedding,
                attention_weights=[],
                retrieved_items=[],
                fusion_metadata={'strategy': 'no_items'}
            )
        
        # Get item embeddings
        item_embeddings = []
        weights = []
        
        for result in retrieval_results:
            dummy_embedding = self._text_to_embedding(result.text)
            item_embeddings.append(dummy_embedding)
            
            # Weight combines similarity and discriminative scores
            weight = 0.6 * result.similarity_score + 0.4 * result.discriminative_score
            weights.append(weight)
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        # Weighted sum
        item_embeddings = np.array(item_embeddings)
        weighted_items = np.sum(item_embeddings * weights.reshape(-1, 1), axis=0)
        
        # Combine with query
        fused_embedding = 0.7 * query_embedding + 0.3 * weighted_items
        
        return FusionResult(
            fused_embedding=fused_embedding,
            attention_weights=weights.tolist(),
            retrieved_items=retrieval_results,
            fusion_metadata={
                'strategy': 'weighted_sum',
                'num_items': len(retrieval_results),
                'weight_entropy': self._compute_entropy(weights)
            }
        )
    
    def fuse_hierarchical(self, 
                         query_embedding: np.ndarray,
                         retrieval_results: List[RetrievalResult]) -> FusionResult:
        """Hierarchical fusion: first by class, then across classes."""
        if not retrieval_results:
            return FusionResult(
                fused_embedding=query_embedding,
                attention_weights=[],
                retrieved_items=[],
                fusion_metadata={'strategy': 'no_items'}
            )
        
        # Group by class
        class_groups = {}
        for i, result in enumerate(retrieval_results):
            if result.label not in class_groups:
                class_groups[result.label] = []
            class_groups[result.label].append((i, result))
        
        # Fuse within each class
        class_embeddings = []
        class_weights = []
        item_to_class_attention = {}
        
        for class_label, items in class_groups.items():
            if len(items) == 1:
                # Single item in class
                _, result = items[0]
                class_emb = self._text_to_embedding(result.text)
                class_weight = result.combined_score
                item_to_class_attention[items[0][0]] = 1.0
            else:
                # Multiple items - use attention within class
                class_results = [item[1] for item in items]
                class_fusion = self.fuse_cross_attention(query_embedding, class_results)
                class_emb = class_fusion.fused_embedding
                class_weight = np.mean([result.combined_score for result in class_results])
                
                # Store within-class attention
                for j, (orig_idx, _) in enumerate(items):
                    item_to_class_attention[orig_idx] = class_fusion.attention_weights[j]
            
            class_embeddings.append(class_emb)
            class_weights.append(class_weight)
        
        # Fuse across classes
        class_embeddings = np.array(class_embeddings)
        class_weights = np.array(class_weights)
        class_weights = class_weights / np.sum(class_weights)
        
        # Final fusion
        cross_class_embedding = np.sum(class_embeddings * class_weights.reshape(-1, 1), axis=0)
        fused_embedding = 0.8 * query_embedding + 0.2 * cross_class_embedding
        
        # Reconstruct item-level attention weights
        final_attention_weights = []
        for i, result in enumerate(retrieval_results):
            class_idx = list(class_groups.keys()).index(result.label)
            within_class_weight = item_to_class_attention.get(i, 0.0)
            across_class_weight = class_weights[class_idx]
            final_weight = within_class_weight * across_class_weight
            final_attention_weights.append(final_weight)
        
        return FusionResult(
            fused_embedding=fused_embedding,
            attention_weights=final_attention_weights,
            retrieved_items=retrieval_results,
            fusion_metadata={
                'strategy': 'hierarchical',
                'num_items': len(retrieval_results),
                'num_classes': len(class_groups),
                'class_distribution': {k: len(v) for k, v in class_groups.items()}
            }
        )
    
    def fuse(self, 
             query_embedding: np.ndarray,
             retrieval_results: List[RetrievalResult]) -> FusionResult:
        """Main fusion method that dispatches to specific strategies."""
        if self.fusion_strategy == "cross_attention":
            return self.fuse_cross_attention(query_embedding, retrieval_results)
        elif self.fusion_strategy == "weighted_sum":
            return self.fuse_weighted_sum(query_embedding, retrieval_results)
        elif self.fusion_strategy == "hierarchical":
            return self.fuse_hierarchical(query_embedding, retrieval_results)
        else:
            raise ValueError(f"Unknown fusion strategy: {self.fusion_strategy}")
    
    def _text_to_embedding(self, text: str) -> np.ndarray:
        """Convert text to embedding (simplified hash-based approach)."""
        # In practice, use proper embeddings from your model
        words = text.lower().split()
        embedding = np.zeros(self.embedding_dim)
        
        for word in words:
            idx = hash(word) % self.embedding_dim
            embedding[idx] += 1.0
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def _compute_entropy(self, weights: np.ndarray) -> float:
        """Compute entropy of attention weights."""
        weights = weights + 1e-8  # Avoid log(0)
        return -np.sum(weights * np.log(weights))
    
    def explain_fusion(self, fusion_result: FusionResult) -> Dict[str, Any]:
        """Explain the fusion process."""
        explanation = {
            'fusion_strategy': fusion_result.fusion_metadata.get('strategy'),
            'num_retrieved_items': len(fusion_result.retrieved_items),
            'attention_analysis': {
                'weights': [round(w, 3) for w in fusion_result.attention_weights],
                'entropy': round(fusion_result.fusion_metadata.get('avg_attention_entropy', 0), 3),
                'max_weight': round(max(fusion_result.attention_weights) if fusion_result.attention_weights else 0, 3),
                'min_weight': round(min(fusion_result.attention_weights) if fusion_result.attention_weights else 0, 3)
            },
            'retrieved_items_summary': []
        }
        
        for i, (item, weight) in enumerate(zip(fusion_result.retrieved_items, fusion_result.attention_weights)):
            item_summary = {
                'rank': i + 1,
                'label': item.label,
                'text_preview': item.text[:50] + '...' if len(item.text) > 50 else item.text,
                'attention_weight': round(weight, 3),
                'similarity_score': round(item.similarity_score, 3),
                'discriminative_score': round(item.discriminative_score, 3)
            }
            explanation['retrieved_items_summary'].append(item_summary)
        
        return explanation
