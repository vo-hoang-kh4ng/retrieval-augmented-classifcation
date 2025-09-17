"""
Enhanced main entry point with Class-aware Retrieval and Latent Fusion.
Benchmarks against baseline and provides comprehensive evaluation.
"""

import argparse
import json
import time
from typing import List, Dict, Any
import numpy as np

from .simple_main import SimpleClassifier, load_sample_data
from .classifiers.enhanced_rac_classifier import EnhancedRACClassifier
from .evaluation.long_tail_metrics import LongTailEvaluator


def load_enhanced_data() -> List[Dict[str, Any]]:
    """Load enhanced dataset with more diverse examples."""
    # Load base data
    base_data = load_sample_data()
    
    # Add more diverse examples for better evaluation
    enhanced_examples = [
        # Software & Technology (541511)
        {"text": "Custom software development and programming services", "label": "541511"},
        {"text": "Mobile application development and deployment", "label": "541511"},
        {"text": "Web development and e-commerce solutions", "label": "541511"},
        {"text": "Database design and management systems", "label": "541511"},
        {"text": "Software consulting and system integration", "label": "541511"},
        
        # Restaurants (722513)
        {"text": "Full-service restaurant with table service", "label": "722513"},
        {"text": "Casual dining restaurant operations", "label": "722513"},
        {"text": "Fine dining establishment with chef service", "label": "722513"},
        
        # Fast Food (722513)
        {"text": "Quick service restaurant and fast food", "label": "722513"},
        {"text": "Drive-through fast food operations", "label": "722513"},
        
        # Coffee Shops (722515)
        {"text": "Coffee shop and specialty beverage service", "label": "722515"},
        {"text": "Espresso bar and cafe operations", "label": "722515"},
        {"text": "Coffee roasting and retail cafe", "label": "722515"},
        
        # Retail Clothing (448140)
        {"text": "Women's clothing and fashion retail", "label": "448140"},
        {"text": "Men's apparel and accessories store", "label": "448140"},
        {"text": "Children's clothing retail operations", "label": "448140"},
        
        # Online Retail (454110)
        {"text": "E-commerce and online retail sales", "label": "454110"},
        {"text": "Internet-based retail operations", "label": "454110"},
        {"text": "Online marketplace and digital sales", "label": "454110"},
        
        # Management Consulting (541611)
        {"text": "Business strategy and management consulting", "label": "541611"},
        {"text": "Organizational development consulting", "label": "541611"},
        {"text": "Process improvement and efficiency consulting", "label": "541611"},
        
        # Long-tail examples (fewer samples)
        {"text": "Specialized medical device manufacturing", "label": "334510"},
        {"text": "Custom jewelry design and manufacturing", "label": "339911"},
        {"text": "Professional photography services", "label": "541921"},
        {"text": "Translation and interpretation services", "label": "541930"},
        {"text": "Environmental consulting services", "label": "541620"},
        {"text": "Architectural design services", "label": "541310"},
        {"text": "Legal services and law practice", "label": "541110"},
        {"text": "Accounting and bookkeeping services", "label": "541211"},
        {"text": "Marketing and advertising agency", "label": "541810"},
        {"text": "Graphic design services", "label": "541430"},
        
        # Very rare examples (1-2 samples each)
        {"text": "Rare book restoration and preservation", "label": "711510"},
        {"text": "Custom musical instrument crafting", "label": "339992"},
        {"text": "Specialized scientific research laboratory", "label": "541712"},
        {"text": "Professional genealogy research services", "label": "541990"},
        {"text": "Custom perfume and fragrance development", "label": "325620"}
    ]
    
    return base_data + enhanced_examples


def benchmark_classifiers(train_data: List[Dict], test_data: List[Dict]) -> Dict[str, Any]:
    """Benchmark baseline vs enhanced classifier."""
    print("🔬 BENCHMARKING CLASSIFIERS")
    print("=" * 60)
    
    # Prepare data
    train_texts = [item["text"] for item in train_data]
    train_labels = [item["label"] for item in train_data]
    test_texts = [item["text"] for item in test_data]
    test_labels = [item["label"] for item in test_data]
    
    results = {}
    
    # 1. Baseline Simple Classifier
    print("\n📊 Training Baseline Classifier...")
    start_time = time.time()
    
    baseline_classifier = SimpleClassifier()
    baseline_classifier.fit(train_texts, train_labels)
    
    baseline_train_time = time.time() - start_time
    
    # Test baseline
    baseline_predictions = []
    baseline_confidences = []
    baseline_test_time = 0
    
    for test_text in test_texts:
        start = time.time()
        result = baseline_classifier.classify(test_text)
        baseline_test_time += time.time() - start
        
        baseline_predictions.append(result["prediction"])
        baseline_confidences.append(result["confidence"])
    
    baseline_accuracy = sum(1 for t, p in zip(test_labels, baseline_predictions) if t == p) / len(test_labels)
    
    results['baseline'] = {
        'accuracy': baseline_accuracy,
        'avg_confidence': np.mean(baseline_confidences),
        'train_time': baseline_train_time,
        'avg_test_time': baseline_test_time / len(test_texts),
        'predictions': baseline_predictions
    }
    
    print(f"✓ Baseline - Accuracy: {baseline_accuracy:.3f}, Train Time: {baseline_train_time:.2f}s")
    
    # 2. Enhanced RAC Classifier
    print("\n🚀 Training Enhanced RAC Classifier...")
    start_time = time.time()
    
    enhanced_classifier = EnhancedRACClassifier(
        embedding_dim=128,
        retrieval_k=8,
        fusion_strategy="hierarchical",
        alpha=0.7,  # Balance similarity vs discriminative
        beta=0.3,   # Class frequency weight
        gamma=0.8,  # Fusion vs similarity in classification
        long_tail_boost=1.5
    )
    enhanced_classifier.fit(train_texts, train_labels)
    
    enhanced_train_time = time.time() - start_time
    
    # Test enhanced classifier
    enhanced_predictions = []
    enhanced_confidences = []
    enhanced_test_time = 0
    enhanced_results = []
    
    for test_text in test_texts:
        start = time.time()
        result = enhanced_classifier.classify(test_text, explain=True)
        enhanced_test_time += time.time() - start
        
        enhanced_predictions.append(result.prediction)
        enhanced_confidences.append(result.confidence)
        enhanced_results.append(result)
    
    enhanced_accuracy = sum(1 for t, p in zip(test_labels, enhanced_predictions) if t == p) / len(test_labels)
    
    results['enhanced'] = {
        'accuracy': enhanced_accuracy,
        'avg_confidence': np.mean(enhanced_confidences),
        'train_time': enhanced_train_time,
        'avg_test_time': enhanced_test_time / len(test_texts),
        'predictions': enhanced_predictions,
        'detailed_results': enhanced_results
    }
    
    print(f"✓ Enhanced - Accuracy: {enhanced_accuracy:.3f}, Train Time: {enhanced_train_time:.2f}s")
    
    # 3. Detailed Long-tail Evaluation
    print("\n📈 Long-tail Evaluation...")
    
    evaluator = LongTailEvaluator()
    
    # Set class frequencies from training data
    class_frequencies = {}
    for label in train_labels:
        class_frequencies[label] = class_frequencies.get(label, 0) + 1
    
    evaluator.set_class_frequencies(class_frequencies)
    
    # Add enhanced predictions for evaluation
    for i, result in enumerate(enhanced_results):
        if i < len(test_labels):
            evaluator.add_prediction(
                true_label=test_labels[i],
                predicted_label=result.prediction,
                class_probabilities=result.class_probabilities,
                retrieval_results=result.retrieval_results,
                fusion_result=result.fusion_result,
                explanation=result.explanation
            )
    
    # Compute comprehensive metrics
    long_tail_metrics = evaluator.evaluate()
    
    results['long_tail_metrics'] = long_tail_metrics
    results['evaluation_report'] = evaluator.generate_report(long_tail_metrics)
    
    # 4. Performance Comparison
    improvement = enhanced_accuracy - baseline_accuracy
    speedup = (baseline_test_time / len(test_texts)) / (enhanced_test_time / len(test_texts))
    
    results['comparison'] = {
        'accuracy_improvement': improvement,
        'relative_improvement': improvement / baseline_accuracy if baseline_accuracy > 0 else 0,
        'speed_ratio': speedup,
        'enhanced_is_better': improvement > 0
    }
    
    print(f"\n📊 COMPARISON SUMMARY")
    print(f"Accuracy Improvement: {improvement:+.3f} ({improvement/baseline_accuracy*100:+.1f}%)")
    print(f"Speed Ratio: {speedup:.2f}x")
    
    return results


def demonstrate_enhanced_features(classifier: EnhancedRACClassifier):
    """Demonstrate enhanced features with example queries."""
    print("\n🎯 ENHANCED FEATURES DEMONSTRATION")
    print("=" * 60)
    
    demo_queries = [
        "Custom software development and AI consulting services",
        "Small family restaurant with delivery service",
        "Specialized environmental impact assessment consulting",
        "Online boutique selling handmade jewelry",
        "Professional translation services for legal documents"
    ]
    
    for i, query in enumerate(demo_queries, 1):
        print(f"\n{i}. Query: '{query}'")
        print("-" * 50)
        
        result = classifier.classify(query, explain=True)
        
        print(f"🎯 Prediction: {result.prediction}")
        print(f"🎯 Confidence: {result.confidence:.3f}")
        print(f"⏱️  Processing Time: {result.processing_time*1000:.1f}ms")
        
        # Show top retrieved items
        print("\n📚 Top Retrieved Items:")
        for j, item in enumerate(result.retrieval_results[:3], 1):
            print(f"  {j}. [{item.label}] {item.text[:60]}...")
            print(f"     Sim: {item.similarity_score:.3f}, Disc: {item.discriminative_score:.3f}")
        
        # Show attention weights
        if result.fusion_result.attention_weights:
            print(f"\n🔍 Attention Weights: {[f'{w:.3f}' for w in result.fusion_result.attention_weights[:3]]}")
        
        # Show class probabilities
        top_classes = sorted(result.class_probabilities.items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"\n📊 Top Classes:")
        for class_name, prob in top_classes:
            print(f"  {class_name}: {prob:.3f}")


def main():
    """Main function for enhanced RAC evaluation."""
    parser = argparse.ArgumentParser(description="Enhanced RAG RAC NAICS Classification")
    parser.add_argument("--mode", choices=["train", "benchmark", "demo"], required=True)
    parser.add_argument("--query", help="Query to classify (for demo mode)")
    parser.add_argument("--save-results", help="Path to save benchmark results")
    parser.add_argument("--fusion-strategy", choices=["cross_attention", "weighted_sum", "hierarchical"], 
                       default="hierarchical", help="Fusion strategy to use")
    
    args = parser.parse_args()
    
    # Load enhanced dataset
    all_data = load_enhanced_data()
    
    # Split data for benchmarking
    np.random.seed(42)  # For reproducible results
    indices = np.random.permutation(len(all_data))
    split_point = int(0.8 * len(all_data))
    
    train_data = [all_data[i] for i in indices[:split_point]]
    test_data = [all_data[i] for i in indices[split_point:]]
    
    if args.mode == "train":
        print("🚀 Training Enhanced RAC Classifier...")
        
        train_texts = [item["text"] for item in train_data]
        train_labels = [item["label"] for item in train_data]
        
        classifier = EnhancedRACClassifier(
            fusion_strategy=args.fusion_strategy,
            long_tail_boost=1.5
        )
        classifier.fit(train_texts, train_labels)
        
        # Save trained model
        classifier.save_model("models/enhanced_rac_classifier.json")
        
        # Test with sample query
        test_query = "Custom web application development and consulting"
        result = classifier.classify(test_query, explain=True)
        
        print(f"\n🎯 Test Classification: '{test_query}'")
        print(f"Prediction: {result.prediction}")
        print(f"Confidence: {result.confidence:.3f}")
        
    elif args.mode == "benchmark":
        print("🔬 Running Comprehensive Benchmark...")
        
        results = benchmark_classifiers(train_data, test_data)
        
        # Print evaluation report
        print("\n" + results['evaluation_report'])
        
        # Save results if requested
        if args.save_results:
            with open(args.save_results, 'w', encoding='utf-8') as f:
                # Convert numpy arrays to lists for JSON serialization
                serializable_results = {}
                for key, value in results.items():
                    if key == 'long_tail_metrics':
                        # Skip complex objects for now
                        continue
                    elif key == 'evaluation_report':
                        serializable_results[key] = value
                    else:
                        serializable_results[key] = value
                
                json.dump(serializable_results, f, indent=2, ensure_ascii=False)
            
            print(f"\n✓ Results saved to {args.save_results}")
    
    elif args.mode == "demo":
        print("🎭 Enhanced RAC Demo Mode...")
        
        # Train classifier
        train_texts = [item["text"] for item in train_data]
        train_labels = [item["label"] for item in train_data]
        
        classifier = EnhancedRACClassifier(
            fusion_strategy=args.fusion_strategy,
            long_tail_boost=1.5
        )
        classifier.fit(train_texts, train_labels)
        
        if args.query:
            # Classify specific query
            result = classifier.classify(args.query, explain=True)
            
            print(f"\n🎯 Classification Result for: '{args.query}'")
            print(f"Prediction: {result.prediction}")
            print(f"Confidence: {result.confidence:.3f}")
            print(f"Processing Time: {result.processing_time*1000:.1f}ms")
            
            # Show detailed explanation
            if result.explanation:
                print(f"\n📋 Detailed Explanation:")
                explanation = result.explanation
                
                print(f"Candidate Classes: {explanation['pipeline_steps']['1_retrieval']['candidate_classes']}")
                print(f"Fusion Strategy: {explanation['pipeline_steps']['2_fusion']['strategy']}")
                print(f"Long-tail Boost Applied: {explanation['pipeline_steps']['3_classification']['long_tail_boost_applied']}")
        else:
            # Run feature demonstration
            demonstrate_enhanced_features(classifier)


if __name__ == "__main__":
    main()
