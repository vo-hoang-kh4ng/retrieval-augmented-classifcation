"""
Quick test script for the enhanced RAC pipeline.
Tests both baseline and enhanced classifiers with sample data.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.rag_rac_naics.enhanced_main import load_enhanced_data, benchmark_classifiers
from src.rag_rac_naics.classifiers.enhanced_rac_classifier import EnhancedRACClassifier
import numpy as np


def quick_test():
    """Quick test of the enhanced pipeline."""
    print("🧪 QUICK TEST: Enhanced RAC Pipeline")
    print("=" * 50)
    
    # Load data
    all_data = load_enhanced_data()
    print(f"✓ Loaded {len(all_data)} examples")
    
    # Split data
    np.random.seed(42)
    indices = np.random.permutation(len(all_data))
    split_point = int(0.8 * len(all_data))
    
    train_data = [all_data[i] for i in indices[:split_point]]
    test_data = [all_data[i] for i in indices[split_point:]]
    
    print(f"✓ Train: {len(train_data)}, Test: {len(test_data)}")
    
    # Quick enhanced classifier test
    print("\n🚀 Testing Enhanced Classifier...")
    
    train_texts = [item["text"] for item in train_data]
    train_labels = [item["label"] for item in train_data]
    
    classifier = EnhancedRACClassifier(
        embedding_dim=64,  # Smaller for quick test
        retrieval_k=5,
        fusion_strategy="cross_attention",
        alpha=0.7,
        beta=0.3,
        gamma=0.8,
        long_tail_boost=1.5
    )
    
    # Train
    classifier.fit(train_texts, train_labels)
    
    # Test with a few examples
    test_queries = [
        "Custom software development services",
        "Restaurant and food delivery",
        "Environmental consulting services",
        "Online retail clothing store"
    ]
    
    print("\n📊 Test Results:")
    for i, query in enumerate(test_queries, 1):
        result = classifier.classify(query, explain=False)
        print(f"{i}. '{query[:40]}...'")
        print(f"   → {result.prediction} (conf: {result.confidence:.3f})")
        print(f"   → Time: {result.processing_time*1000:.1f}ms")
    
    # Performance stats
    stats = classifier.get_performance_summary()
    print(f"\n📈 Performance Summary:")
    print(f"Total predictions: {stats['total_predictions']}")
    print(f"Avg processing time: {stats['avg_processing_time']*1000:.1f}ms")
    print(f"Long-tail classes: {len(stats.get('class_distribution', {}).get('long_tail_classes', []))}")
    
    print("\n✅ Enhanced pipeline test completed successfully!")


def detailed_example():
    """Show detailed example with explanations."""
    print("\n🔍 DETAILED EXAMPLE")
    print("=" * 50)
    
    # Load and prepare data
    all_data = load_enhanced_data()
    train_texts = [item["text"] for item in all_data[:50]]  # Use first 50 for quick demo
    train_labels = [item["label"] for item in all_data[:50]]
    
    # Create classifier
    classifier = EnhancedRACClassifier(
        embedding_dim=64,
        retrieval_k=5,
        fusion_strategy="hierarchical",
        long_tail_boost=2.0
    )
    
    classifier.fit(train_texts, train_labels)
    
    # Test with detailed explanation
    query = "Specialized AI consulting and machine learning services"
    print(f"\n🎯 Query: '{query}'")
    
    result = classifier.classify(query, explain=True)
    
    print(f"\n📊 Results:")
    print(f"Prediction: {result.prediction}")
    print(f"Confidence: {result.confidence:.3f}")
    print(f"Processing Time: {result.processing_time*1000:.1f}ms")
    
    print(f"\n🔍 Retrieved Items:")
    for i, item in enumerate(result.retrieval_results[:3], 1):
        print(f"{i}. [{item.label}] {item.text[:50]}...")
        print(f"   Similarity: {item.similarity_score:.3f}")
        print(f"   Discriminative: {item.discriminative_score:.3f}")
        print(f"   Combined: {item.combined_score:.3f}")
    
    print(f"\n🧠 Fusion Info:")
    print(f"Strategy: {result.fusion_result.fusion_metadata.get('strategy', 'N/A')}")
    print(f"Attention weights: {[f'{w:.3f}' for w in result.fusion_result.attention_weights[:3]]}")
    
    print(f"\n📈 Top Class Probabilities:")
    sorted_probs = sorted(result.class_probabilities.items(), key=lambda x: x[1], reverse=True)
    for class_name, prob in sorted_probs[:3]:
        print(f"  {class_name}: {prob:.3f}")


if __name__ == "__main__":
    try:
        quick_test()
        detailed_example()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
