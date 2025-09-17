"""Simple main entry point without heavy dependencies."""

import argparse
import os
import json
from typing import List, Dict, Any

# Simple implementations without numpy/sklearn
class SimpleEmbeddings:
    """Hash-based embeddings without numpy dependency."""
    
    def __init__(self, dim: int = 128):
        self.dim = dim
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate simple hash-based embeddings."""
        embeddings = []
        for text in texts:
            vec = [0.0] * self.dim
            words = (text or "").lower().split()
            for word in words:
                idx = hash(word) % self.dim
                vec[idx] += 1.0
            
            # Simple normalization
            norm = sum(v * v for v in vec) ** 0.5
            if norm > 0:
                vec = [v / norm for v in vec]
            
            embeddings.append(vec)
        return embeddings

class SimpleClassifier:
    """Simple classifier using cosine similarity."""
    
    def __init__(self):
        self.embeddings = SimpleEmbeddings()
        self.texts = []
        self.labels = []
        self.text_embeddings = []
    
    def fit(self, texts: List[str], labels: List[str]):
        """Train the classifier."""
        self.texts = texts
        self.labels = labels
        self.text_embeddings = self.embeddings.get_embeddings(texts)
        print(f"✓ Trained classifier with {len(texts)} examples")
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def classify(self, text: str, k: int = 3) -> Dict[str, Any]:
        """Classify text and return results."""
        if not self.texts:
            return {"prediction": "UNKNOWN", "confidence": 0.0, "examples": []}
        
        query_embedding = self.embeddings.get_embeddings([text])[0]
        
        # Calculate similarities
        similarities = []
        for i, text_emb in enumerate(self.text_embeddings):
            sim = self.cosine_similarity(query_embedding, text_emb)
            similarities.append((sim, i))
        
        # Sort by similarity
        similarities.sort(reverse=True)
        
        # Get top k
        top_k = similarities[:k]
        
        # Get most common label
        top_labels = [self.labels[idx] for _, idx in top_k]
        prediction = max(set(top_labels), key=top_labels.count) if top_labels else "UNKNOWN"
        
        # Calculate confidence
        confidence = top_k[0][0] if top_k else 0.0
        
        # Get examples
        examples = [
            {
                "text": self.texts[idx],
                "label": self.labels[idx], 
                "score": sim
            }
            for sim, idx in top_k
        ]
        
        return {
            "prediction": prediction,
            "confidence": confidence,
            "examples": examples
        }

def load_sample_data() -> List[Dict[str, Any]]:
    """Load sample data."""
    data_path = "data/sample_data.json"
    if os.path.exists(data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        # Fallback sample data
        return [
            {"text": "Software development and programming services", "label": "541511"},
            {"text": "Web design and development", "label": "541511"},
            {"text": "Mobile app development", "label": "541511"},
            {"text": "Restaurant and food service", "label": "722513"},
            {"text": "Fast food restaurant", "label": "722513"},
            {"text": "Coffee shop and cafe services", "label": "722515"},
            {"text": "Retail clothing store", "label": "448140"},
            {"text": "Online retail sales", "label": "454110"},
            {"text": "Consulting services", "label": "541611"},
            {"text": "Management consulting", "label": "541611"}
        ]

def main():
    parser = argparse.ArgumentParser(description="Simple RAG RAC NAICS Classification")
    parser.add_argument("--mode", choices=["train", "classify"], required=True)
    parser.add_argument("--query", help="Query to classify")
    
    args = parser.parse_args()
    
    # Load data
    sample_data = load_sample_data()
    texts = [item["text"] for item in sample_data]
    labels = [item["label"] for item in sample_data]
    
    # Create classifier
    classifier = SimpleClassifier()
    
    if args.mode == "train":
        print("Training simple classifier...")
        classifier.fit(texts, labels)
        print(f"✓ Training completed with {len(texts)} examples")
        
        # Test with a sample query
        test_query = "Custom web application development"
        result = classifier.classify(test_query)
        print(f"\nTest classification for: '{test_query}'")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print("Top examples:")
        for i, ex in enumerate(result['examples'], 1):
            print(f"  {i}. [{ex['label']}] {ex['text']} (score: {ex['score']:.3f})")
    
    elif args.mode == "classify":
        if not args.query:
            print("Error: --query is required for classify mode")
            return
        
        print("Training classifier...")
        classifier.fit(texts, labels)
        
        print(f"Classifying: '{args.query}'")
        result = classifier.classify(args.query)
        
        print(f"\nClassification Results:")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print("Retrieved Examples:")
        for i, ex in enumerate(result['examples'], 1):
            print(f"  {i}. [{ex['label']}] {ex['text']} (score: {ex['score']:.3f})")

if __name__ == "__main__":
    main()
