"""
Simple enhanced main without numpy dependency.
Demonstrates key features of the enhanced pipeline.
"""

import argparse
import json
import time
import random
from typing import List, Dict, Any

from .simple_main import SimpleClassifier, load_sample_data


class SimpleEnhancedClassifier:
    """
    Simplified version of enhanced classifier without heavy dependencies.
    Demonstrates key concepts: discriminative retrieval and fusion.
    """
    
    def __init__(self, 
                 retrieval_k: int = 8,
                 alpha: float = 0.7,  # Similarity vs discriminative balance
                 beta: float = 0.3,   # Class frequency weight
                 long_tail_boost: float = 1.5):
        self.retrieval_k = retrieval_k
        self.alpha = alpha
        self.beta = beta
        self.long_tail_boost = long_tail_boost
        
        # Base classifier
        self.base_classifier = SimpleClassifier()
        
        # Enhanced components
        self.class_stats = {}
        self.long_tail_classes = set()
        self.training_texts = []
        self.training_labels = []
    
    def fit(self, texts: List[str], labels: List[str]):
        """Train the enhanced classifier."""
        print("Training Simple Enhanced Classifier...")
        
        self.training_texts = texts
        self.training_labels = labels
        
        # Train base classifier
        self.base_classifier.fit(texts, labels)
        
        # Compute class statistics
        self._compute_class_stats()
        
        print(f"✓ Trained with {len(texts)} examples")
        print(f"✓ Found {len(self.class_stats)} classes")
        print(f"✓ Long-tail classes: {len(self.long_tail_classes)}")
    
    def _compute_class_stats(self):
        """Compute class frequency statistics."""
        class_counts = {}
        for label in self.training_labels:
            class_counts[label] = class_counts.get(label, 0) + 1
        
        total_samples = len(self.training_labels)
        long_tail_threshold = max(1, int(total_samples * 0.05))  # < 5% of data
        
        for class_label, count in class_counts.items():
            relative_freq = count / total_samples
            is_long_tail = count < long_tail_threshold
            
            self.class_stats[class_label] = {
                'count': count,
                'relative_frequency': relative_freq,
                'is_long_tail': is_long_tail
            }
            
            if is_long_tail:
                self.long_tail_classes.add(class_label)
    
    def _enhanced_retrieval(self, query: str) -> List[Dict[str, Any]]:
        """Enhanced retrieval with discriminative scoring."""
        # Get base retrieval results
        base_results = self.base_classifier.classify(query, k=self.retrieval_k * 2)
        
        # Add discriminative scoring
        enhanced_results = []
        
        for example in base_results["examples"]:
            # Simulate discriminative score based on class diversity
            discriminative_score = self._compute_discriminative_score(
                example["label"], 
                [ex["label"] for ex in base_results["examples"]]
            )
            
            # Class frequency weight
            freq_weight = self._compute_frequency_weight(example["label"])
            
            # Combined score
            combined_score = (
                self.alpha * example["score"] + 
                (1 - self.alpha) * discriminative_score +
                self.beta * freq_weight
            )
            
            enhanced_example = {
                "text": example["text"],
                "label": example["label"],
                "similarity_score": example["score"],
                "discriminative_score": discriminative_score,
                "frequency_weight": freq_weight,
                "combined_score": combined_score,
                "is_long_tail": example["label"] in self.long_tail_classes
            }
            
            enhanced_results.append(enhanced_example)
        
        # Sort by combined score and return top k
        enhanced_results.sort(key=lambda x: x["combined_score"], reverse=True)
        return enhanced_results[:self.retrieval_k]
    
    def _compute_discriminative_score(self, item_label: str, all_labels: List[str]) -> float:
        """Compute discriminative score (simplified)."""
        # Higher score if this label is rare among retrieved items
        label_count = all_labels.count(item_label)
        total_items = len(all_labels)
        
        if total_items == 0:
            return 0.0
        
        # Inverse frequency in retrieved set
        item_frequency = label_count / total_items
        discriminative_score = 1.0 - item_frequency
        
        return discriminative_score
    
    def _compute_frequency_weight(self, class_label: str) -> float:
        """Compute frequency-based weight for long-tail support."""
        if class_label not in self.class_stats:
            return 1.0
        
        stats = self.class_stats[class_label]
        
        if stats["is_long_tail"]:
            # Boost long-tail classes
            return 1.0 + (1.0 - stats["relative_frequency"]) * 0.5
        else:
            return 1.0
    
    def _fusion_scoring(self, query: str, retrieved_items: List[Dict]) -> Dict[str, float]:
        """Simplified fusion scoring."""
        class_scores = {}
        
        # Weighted voting based on enhanced scores
        for item in retrieved_items:
            label = item["label"]
            if label not in class_scores:
                class_scores[label] = 0.0
            
            # Weight combines multiple factors
            vote_weight = (
                0.4 * item["similarity_score"] +
                0.4 * item["discriminative_score"] +
                0.2 * item["frequency_weight"]
            )
            
            class_scores[label] += vote_weight
        
        # Apply long-tail boost
        for label in class_scores:
            if label in self.long_tail_classes:
                class_scores[label] *= self.long_tail_boost
        
        # Normalize to probabilities
        total_score = sum(class_scores.values())
        if total_score > 0:
            class_scores = {k: v / total_score for k, v in class_scores.items()}
        
        return class_scores
    
    def classify(self, query: str, explain: bool = True) -> Dict[str, Any]:
        """Enhanced classification with detailed results."""
        start_time = time.time()
        
        # Step 1: Enhanced retrieval
        retrieved_items = self._enhanced_retrieval(query)
        
        # Step 2: Fusion scoring
        class_probabilities = self._fusion_scoring(query, retrieved_items)
        
        # Step 3: Make prediction
        if class_probabilities:
            prediction = max(class_probabilities.items(), key=lambda x: x[1])[0]
            confidence = class_probabilities[prediction]
        else:
            prediction = "UNKNOWN"
            confidence = 0.0
        
        processing_time = time.time() - start_time
        
        result = {
            "query": query,
            "prediction": prediction,
            "confidence": confidence,
            "processing_time": processing_time,
            "class_probabilities": class_probabilities,
            "retrieved_items": retrieved_items
        }
        
        if explain:
            result["explanation"] = self._generate_explanation(query, retrieved_items, class_probabilities)
        
        return result
    
    def _generate_explanation(self, query: str, retrieved_items: List[Dict], class_probabilities: Dict[str, float]) -> Dict[str, Any]:
        """Generate explanation of the classification process."""
        
        # Analyze retrieval quality
        retrieval_analysis = {
            "num_retrieved": len(retrieved_items),
            "class_diversity": len(set(item["label"] for item in retrieved_items)),
            "long_tail_coverage": sum(1 for item in retrieved_items if item["is_long_tail"]),
            "avg_discriminative_score": sum(item["discriminative_score"] for item in retrieved_items) / len(retrieved_items) if retrieved_items else 0,
            "top_items": [
                {
                    "text": item["text"][:60] + "..." if len(item["text"]) > 60 else item["text"],
                    "label": item["label"],
                    "similarity": round(item["similarity_score"], 3),
                    "discriminative": round(item["discriminative_score"], 3),
                    "combined": round(item["combined_score"], 3),
                    "is_long_tail": item["is_long_tail"]
                }
                for item in retrieved_items[:3]
            ]
        }
        
        # Analyze classification
        sorted_probs = sorted(class_probabilities.items(), key=lambda x: x[1], reverse=True)
        classification_analysis = {
            "top_classes": [
                {
                    "class": class_name,
                    "probability": round(prob, 3),
                    "is_long_tail": class_name in self.long_tail_classes,
                    "training_count": self.class_stats.get(class_name, {}).get("count", 0)
                }
                for class_name, prob in sorted_probs[:5]
            ],
            "long_tail_boost_applied": any(cls in self.long_tail_classes for cls, _ in sorted_probs[:3])
        }
        
        return {
            "retrieval_analysis": retrieval_analysis,
            "classification_analysis": classification_analysis,
            "pipeline_features": {
                "discriminative_retrieval": True,
                "long_tail_support": True,
                "fusion_scoring": True
            }
        }


def load_enhanced_sample_data() -> List[Dict[str, Any]]:
    """Load enhanced sample data with more examples."""
    base_data = load_sample_data()
    
    # Add more examples for better demonstration
    additional_examples = [
        {"text": "AI and machine learning consulting services", "label": "541511"},
        {"text": "Custom mobile app development", "label": "541511"},
        {"text": "Fine dining restaurant operations", "label": "722513"},
        {"text": "Coffee shop and bakery services", "label": "722515"},
        {"text": "Online fashion retail store", "label": "454110"},
        {"text": "Business strategy consulting", "label": "541611"},
        {"text": "Environmental impact assessment", "label": "541620"},
        {"text": "Professional photography services", "label": "541921"},
        {"text": "Legal document translation services", "label": "541930"},
        {"text": "Custom jewelry design", "label": "339911"},
    ]
    
    return base_data + additional_examples


def benchmark_simple_enhanced():
    """Simple benchmark between baseline and enhanced."""
    print("🔬 SIMPLE BENCHMARK")
    print("=" * 50)
    
    # Load data
    all_data = load_enhanced_sample_data()
    
    # Split data (simple split)
    random.seed(42)
    random.shuffle(all_data)
    split_point = int(0.8 * len(all_data))
    
    train_data = all_data[:split_point]
    test_data = all_data[split_point:]
    
    train_texts = [item["text"] for item in train_data]
    train_labels = [item["label"] for item in train_data]
    test_texts = [item["text"] for item in test_data]
    test_labels = [item["label"] for item in test_data]
    
    print(f"Train: {len(train_data)}, Test: {len(test_data)}")
    
    # Test baseline
    print("\n📊 Baseline Classifier:")
    baseline = SimpleClassifier()
    baseline.fit(train_texts, train_labels)
    
    baseline_correct = 0
    baseline_time = 0
    
    for text, true_label in zip(test_texts, test_labels):
        start = time.time()
        result = baseline.classify(text)
        baseline_time += time.time() - start
        
        if result["prediction"] == true_label:
            baseline_correct += 1
    
    baseline_accuracy = baseline_correct / len(test_texts)
    baseline_avg_time = baseline_time / len(test_texts)
    
    print(f"Accuracy: {baseline_accuracy:.3f}")
    print(f"Avg Time: {baseline_avg_time*1000:.1f}ms")
    
    # Test enhanced
    print("\n🚀 Enhanced Classifier:")
    enhanced = SimpleEnhancedClassifier(
        retrieval_k=6,
        alpha=0.7,
        beta=0.3,
        long_tail_boost=1.5
    )
    enhanced.fit(train_texts, train_labels)
    
    enhanced_correct = 0
    enhanced_time = 0
    
    for text, true_label in zip(test_texts, test_labels):
        start = time.time()
        result = enhanced.classify(text, explain=False)
        enhanced_time += time.time() - start
        
        if result["prediction"] == true_label:
            enhanced_correct += 1
    
    enhanced_accuracy = enhanced_correct / len(test_texts)
    enhanced_avg_time = enhanced_time / len(test_texts)
    
    print(f"Accuracy: {enhanced_accuracy:.3f}")
    print(f"Avg Time: {enhanced_avg_time*1000:.1f}ms")
    
    # Comparison
    improvement = enhanced_accuracy - baseline_accuracy
    print(f"\n📈 Improvement: {improvement:+.3f} ({improvement/baseline_accuracy*100:+.1f}%)")
    
    return {
        "baseline_accuracy": baseline_accuracy,
        "enhanced_accuracy": enhanced_accuracy,
        "improvement": improvement
    }


def demo_enhanced_features():
    """Demonstrate enhanced features."""
    print("\n🎯 ENHANCED FEATURES DEMO")
    print("=" * 50)
    
    # Load and train
    all_data = load_enhanced_sample_data()
    texts = [item["text"] for item in all_data]
    labels = [item["label"] for item in all_data]
    
    classifier = SimpleEnhancedClassifier(
        retrieval_k=5,
        long_tail_boost=2.0
    )
    classifier.fit(texts, labels)
    
    # Demo queries
    demo_queries = [
        "AI consulting and machine learning services",
        "Small family restaurant with takeout",
        "Environmental consulting for businesses",
        "Custom software development company"
    ]
    
    for i, query in enumerate(demo_queries, 1):
        print(f"\n{i}. Query: '{query}'")
        print("-" * 40)
        
        result = classifier.classify(query, explain=True)
        
        print(f"🎯 Prediction: {result['prediction']}")
        print(f"🎯 Confidence: {result['confidence']:.3f}")
        print(f"⏱️  Time: {result['processing_time']*1000:.1f}ms")
        
        # Show retrieval analysis
        retrieval = result["explanation"]["retrieval_analysis"]
        print(f"\n📚 Retrieval:")
        print(f"  Items: {retrieval['num_retrieved']}")
        print(f"  Classes: {retrieval['class_diversity']}")
        print(f"  Long-tail: {retrieval['long_tail_coverage']}")
        print(f"  Avg Discriminative: {retrieval['avg_discriminative_score']:.3f}")
        
        # Show top items
        print(f"\n🔍 Top Retrieved:")
        for j, item in enumerate(retrieval["top_items"], 1):
            print(f"  {j}. [{item['label']}] {item['text']}")
            print(f"     Sim: {item['similarity']}, Disc: {item['discriminative']}")
        
        # Show classification
        classification = result["explanation"]["classification_analysis"]
        print(f"\n📊 Top Classes:")
        for cls_info in classification["top_classes"][:3]:
            long_tail_marker = " (LT)" if cls_info["is_long_tail"] else ""
            print(f"  {cls_info['class']}: {cls_info['probability']:.3f}{long_tail_marker}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Simple Enhanced RAG RAC")
    parser.add_argument("--mode", choices=["demo", "benchmark", "classify"], required=True)
    parser.add_argument("--query", help="Query to classify")
    
    args = parser.parse_args()
    
    if args.mode == "demo":
        demo_enhanced_features()
        
    elif args.mode == "benchmark":
        benchmark_simple_enhanced()
        
    elif args.mode == "classify":
        if not args.query:
            print("Error: --query required for classify mode")
            return
        
        # Quick classification
        all_data = load_enhanced_sample_data()
        texts = [item["text"] for item in all_data]
        labels = [item["label"] for item in all_data]
        
        classifier = SimpleEnhancedClassifier()
        classifier.fit(texts, labels)
        
        result = classifier.classify(args.query, explain=True)
        
        print(f"Query: '{args.query}'")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.3f}")
        
        # Show explanation
        if "explanation" in result:
            print(f"\nLong-tail boost applied: {result['explanation']['classification_analysis']['long_tail_boost_applied']}")


if __name__ == "__main__":
    main()
