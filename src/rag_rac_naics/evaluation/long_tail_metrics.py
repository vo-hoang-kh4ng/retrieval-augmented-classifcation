"""
Evaluation metrics specifically designed for long-tail classification.
Focuses on few-shot and zero-shot performance analysis.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict, Counter
import json
from dataclasses import dataclass
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


@dataclass
class LongTailMetrics:
    """Comprehensive metrics for long-tail classification."""
    
    # Overall metrics
    overall_accuracy: float
    macro_f1: float
    weighted_f1: float
    
    # Long-tail specific metrics
    head_accuracy: float  # Frequent classes (>5% of data)
    tail_accuracy: float  # Rare classes (<5% of data)
    few_shot_accuracy: float  # Classes with 1-10 examples
    zero_shot_accuracy: float  # Classes not seen in training
    
    # Class-wise breakdown
    class_metrics: Dict[str, Dict[str, float]]
    
    # Retrieval quality metrics
    retrieval_precision_at_k: Dict[int, float]
    discriminative_score_correlation: float
    
    # Fusion quality metrics
    attention_entropy_stats: Dict[str, float]
    fusion_effectiveness: float
    
    # Performance by class frequency
    frequency_bins: Dict[str, Dict[str, float]]


class LongTailEvaluator:
    """Evaluator for long-tail classification performance."""
    
    def __init__(self, 
                 frequency_thresholds: Dict[str, float] = None,
                 k_values: List[int] = None):
        """
        Args:
            frequency_thresholds: Thresholds for defining head/tail classes
            k_values: K values for precision@k evaluation
        """
        self.frequency_thresholds = frequency_thresholds or {
            'head': 0.05,    # Classes with >5% of data
            'tail': 0.01,    # Classes with <1% of data
            'few_shot': 10,  # Classes with <=10 examples (absolute count)
            'zero_shot': 0   # Classes with 0 examples in training
        }
        
        self.k_values = k_values or [1, 3, 5, 10]
        
        # Store evaluation data
        self.predictions = []
        self.true_labels = []
        self.class_frequencies = {}
        self.retrieval_results = []
        self.fusion_results = []
        self.explanations = []
    
    def add_prediction(self, 
                      true_label: str,
                      predicted_label: str,
                      class_probabilities: Dict[str, float],
                      retrieval_results: List[Any] = None,
                      fusion_result: Any = None,
                      explanation: Dict[str, Any] = None):
        """Add a prediction for evaluation."""
        self.predictions.append(predicted_label)
        self.true_labels.append(true_label)
        
        if retrieval_results:
            self.retrieval_results.append(retrieval_results)
        
        if fusion_result:
            self.fusion_results.append(fusion_result)
            
        if explanation:
            self.explanations.append(explanation)
    
    def set_class_frequencies(self, class_frequencies: Dict[str, int]):
        """Set class frequency information from training data."""
        self.class_frequencies = class_frequencies
        total_samples = sum(class_frequencies.values())
        
        # Categorize classes
        self.head_classes = set()
        self.tail_classes = set()
        self.few_shot_classes = set()
        
        for class_name, count in class_frequencies.items():
            relative_freq = count / total_samples
            
            if relative_freq > self.frequency_thresholds['head']:
                self.head_classes.add(class_name)
            elif relative_freq < self.frequency_thresholds['tail']:
                self.tail_classes.add(class_name)
            
            if count <= self.frequency_thresholds['few_shot']:
                self.few_shot_classes.add(class_name)
    
    def _compute_accuracy_by_group(self, group_classes: set) -> float:
        """Compute accuracy for a specific group of classes."""
        if not group_classes:
            return 0.0
        
        correct = 0
        total = 0
        
        for true_label, pred_label in zip(self.true_labels, self.predictions):
            if true_label in group_classes:
                total += 1
                if true_label == pred_label:
                    correct += 1
        
        return correct / total if total > 0 else 0.0
    
    def _compute_class_wise_metrics(self) -> Dict[str, Dict[str, float]]:
        """Compute detailed metrics for each class."""
        class_metrics = {}
        
        # Get unique classes
        all_classes = set(self.true_labels + self.predictions)
        
        for class_name in all_classes:
            # True positives, false positives, false negatives
            tp = sum(1 for t, p in zip(self.true_labels, self.predictions) 
                    if t == class_name and p == class_name)
            fp = sum(1 for t, p in zip(self.true_labels, self.predictions) 
                    if t != class_name and p == class_name)
            fn = sum(1 for t, p in zip(self.true_labels, self.predictions) 
                    if t == class_name and p != class_name)
            
            # Compute metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            # Class characteristics
            frequency = self.class_frequencies.get(class_name, 0)
            total_samples = sum(self.class_frequencies.values()) if self.class_frequencies else 1
            relative_frequency = frequency / total_samples
            
            class_metrics[class_name] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'support': sum(1 for t in self.true_labels if t == class_name),
                'frequency': frequency,
                'relative_frequency': relative_frequency,
                'is_head': class_name in self.head_classes,
                'is_tail': class_name in self.tail_classes,
                'is_few_shot': class_name in self.few_shot_classes
            }
        
        return class_metrics
    
    def _compute_retrieval_metrics(self) -> Dict[str, float]:
        """Compute retrieval quality metrics."""
        if not self.retrieval_results:
            return {}
        
        precision_at_k = {}
        discriminative_scores = []
        
        for k in self.k_values:
            correct_retrievals = 0
            total_queries = 0
            
            for i, retrieval_result in enumerate(self.retrieval_results):
                if i < len(self.true_labels):
                    true_label = self.true_labels[i]
                    
                    # Check top-k retrievals
                    top_k_labels = [r.label for r in retrieval_result[:k]]
                    if true_label in top_k_labels:
                        correct_retrievals += 1
                    total_queries += 1
                    
                    # Collect discriminative scores
                    for r in retrieval_result[:k]:
                        discriminative_scores.append(r.discriminative_score)
            
            precision_at_k[k] = correct_retrievals / total_queries if total_queries > 0 else 0.0
        
        # Correlation between discriminative scores and relevance
        discriminative_correlation = np.std(discriminative_scores) if discriminative_scores else 0.0
        
        return {
            'precision_at_k': precision_at_k,
            'discriminative_score_correlation': discriminative_correlation,
            'avg_discriminative_score': np.mean(discriminative_scores) if discriminative_scores else 0.0
        }
    
    def _compute_fusion_metrics(self) -> Dict[str, float]:
        """Compute fusion quality metrics."""
        if not self.fusion_results:
            return {}
        
        attention_entropies = []
        fusion_effectiveness_scores = []
        
        for fusion_result in self.fusion_results:
            # Attention entropy (higher = more distributed attention)
            if hasattr(fusion_result, 'attention_weights') and fusion_result.attention_weights:
                weights = np.array(fusion_result.attention_weights)
                weights = weights + 1e-8  # Avoid log(0)
                entropy = -np.sum(weights * np.log(weights))
                attention_entropies.append(entropy)
            
            # Fusion effectiveness (how much fusion changes the representation)
            if hasattr(fusion_result, 'fusion_metadata'):
                metadata = fusion_result.fusion_metadata
                if 'gate_weights_mean' in metadata:
                    fusion_effectiveness_scores.append(metadata['gate_weights_mean'])
        
        return {
            'attention_entropy_stats': {
                'mean': np.mean(attention_entropies) if attention_entropies else 0.0,
                'std': np.std(attention_entropies) if attention_entropies else 0.0,
                'min': np.min(attention_entropies) if attention_entropies else 0.0,
                'max': np.max(attention_entropies) if attention_entropies else 0.0
            },
            'fusion_effectiveness': np.mean(fusion_effectiveness_scores) if fusion_effectiveness_scores else 0.0
        }
    
    def _compute_frequency_bin_metrics(self) -> Dict[str, Dict[str, float]]:
        """Compute metrics by class frequency bins."""
        if not self.class_frequencies:
            return {}
        
        # Define frequency bins
        total_samples = sum(self.class_frequencies.values())
        bins = {
            'very_frequent': [],    # >10% of data
            'frequent': [],         # 5-10% of data  
            'medium': [],           # 1-5% of data
            'rare': [],             # 0.1-1% of data
            'very_rare': []         # <0.1% of data
        }
        
        # Categorize classes into bins
        for class_name, count in self.class_frequencies.items():
            relative_freq = count / total_samples
            
            if relative_freq > 0.1:
                bins['very_frequent'].append(class_name)
            elif relative_freq > 0.05:
                bins['frequent'].append(class_name)
            elif relative_freq > 0.01:
                bins['medium'].append(class_name)
            elif relative_freq > 0.001:
                bins['rare'].append(class_name)
            else:
                bins['very_rare'].append(class_name)
        
        # Compute metrics for each bin
        bin_metrics = {}
        for bin_name, classes in bins.items():
            if classes:
                accuracy = self._compute_accuracy_by_group(set(classes))
                
                # Compute average F1 for classes in this bin
                f1_scores = []
                for class_name in classes:
                    class_f1 = 0.0  # Would compute from confusion matrix
                    # Simplified calculation
                    tp = sum(1 for t, p in zip(self.true_labels, self.predictions) 
                            if t == class_name and p == class_name)
                    fp = sum(1 for t, p in zip(self.true_labels, self.predictions) 
                            if t != class_name and p == class_name)
                    fn = sum(1 for t, p in zip(self.true_labels, self.predictions) 
                            if t == class_name and p != class_name)
                    
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                    f1_scores.append(f1)
                
                bin_metrics[bin_name] = {
                    'accuracy': accuracy,
                    'avg_f1': np.mean(f1_scores) if f1_scores else 0.0,
                    'num_classes': len(classes),
                    'total_samples': sum(self.class_frequencies[c] for c in classes)
                }
        
        return bin_metrics
    
    def evaluate(self) -> LongTailMetrics:
        """Compute comprehensive long-tail metrics."""
        if not self.predictions or not self.true_labels:
            raise ValueError("No predictions to evaluate")
        
        # Overall metrics
        overall_accuracy = sum(1 for t, p in zip(self.true_labels, self.predictions) if t == p) / len(self.predictions)
        
        # Class-wise metrics
        class_metrics = self._compute_class_wise_metrics()
        
        # Compute macro and weighted F1
        f1_scores = [metrics['f1_score'] for metrics in class_metrics.values()]
        supports = [metrics['support'] for metrics in class_metrics.values()]
        
        macro_f1 = np.mean(f1_scores) if f1_scores else 0.0
        weighted_f1 = np.average(f1_scores, weights=supports) if f1_scores and supports else 0.0
        
        # Group-specific accuracies
        head_accuracy = self._compute_accuracy_by_group(self.head_classes)
        tail_accuracy = self._compute_accuracy_by_group(self.tail_classes)
        few_shot_accuracy = self._compute_accuracy_by_group(self.few_shot_classes)
        
        # Zero-shot accuracy (classes not in training)
        zero_shot_classes = set(self.true_labels) - set(self.class_frequencies.keys())
        zero_shot_accuracy = self._compute_accuracy_by_group(zero_shot_classes)
        
        # Retrieval metrics
        retrieval_metrics = self._compute_retrieval_metrics()
        
        # Fusion metrics
        fusion_metrics = self._compute_fusion_metrics()
        
        # Frequency bin metrics
        frequency_bin_metrics = self._compute_frequency_bin_metrics()
        
        return LongTailMetrics(
            overall_accuracy=overall_accuracy,
            macro_f1=macro_f1,
            weighted_f1=weighted_f1,
            head_accuracy=head_accuracy,
            tail_accuracy=tail_accuracy,
            few_shot_accuracy=few_shot_accuracy,
            zero_shot_accuracy=zero_shot_accuracy,
            class_metrics=class_metrics,
            retrieval_precision_at_k=retrieval_metrics.get('precision_at_k', {}),
            discriminative_score_correlation=retrieval_metrics.get('discriminative_score_correlation', 0.0),
            attention_entropy_stats=fusion_metrics.get('attention_entropy_stats', {}),
            fusion_effectiveness=fusion_metrics.get('fusion_effectiveness', 0.0),
            frequency_bins=frequency_bin_metrics
        )
    
    def generate_report(self, metrics: LongTailMetrics) -> str:
        """Generate a comprehensive evaluation report."""
        report = []
        report.append("=" * 80)
        report.append("LONG-TAIL CLASSIFICATION EVALUATION REPORT")
        report.append("=" * 80)
        
        # Overall Performance
        report.append("\n📊 OVERALL PERFORMANCE")
        report.append("-" * 40)
        report.append(f"Overall Accuracy:     {metrics.overall_accuracy:.3f}")
        report.append(f"Macro F1 Score:       {metrics.macro_f1:.3f}")
        report.append(f"Weighted F1 Score:    {metrics.weighted_f1:.3f}")
        
        # Long-tail Performance
        report.append("\n🎯 LONG-TAIL PERFORMANCE")
        report.append("-" * 40)
        report.append(f"Head Classes Accuracy:     {metrics.head_accuracy:.3f}")
        report.append(f"Tail Classes Accuracy:     {metrics.tail_accuracy:.3f}")
        report.append(f"Few-shot Accuracy:         {metrics.few_shot_accuracy:.3f}")
        report.append(f"Zero-shot Accuracy:        {metrics.zero_shot_accuracy:.3f}")
        
        # Class Distribution Analysis
        report.append("\n📈 CLASS DISTRIBUTION ANALYSIS")
        report.append("-" * 40)
        head_classes = [c for c, m in metrics.class_metrics.items() if m.get('is_head', False)]
        tail_classes = [c for c, m in metrics.class_metrics.items() if m.get('is_tail', False)]
        few_shot_classes = [c for c, m in metrics.class_metrics.items() if m.get('is_few_shot', False)]
        
        report.append(f"Total Classes:         {len(metrics.class_metrics)}")
        report.append(f"Head Classes:          {len(head_classes)}")
        report.append(f"Tail Classes:          {len(tail_classes)}")
        report.append(f"Few-shot Classes:      {len(few_shot_classes)}")
        
        # Retrieval Quality
        if metrics.retrieval_precision_at_k:
            report.append("\n🔍 RETRIEVAL QUALITY")
            report.append("-" * 40)
            for k, precision in metrics.retrieval_precision_at_k.items():
                report.append(f"Precision@{k}:             {precision:.3f}")
            report.append(f"Discriminative Score Corr: {metrics.discriminative_score_correlation:.3f}")
        
        # Fusion Quality
        if metrics.attention_entropy_stats:
            report.append("\n🔗 FUSION QUALITY")
            report.append("-" * 40)
            entropy_stats = metrics.attention_entropy_stats
            report.append(f"Attention Entropy (mean):  {entropy_stats.get('mean', 0):.3f}")
            report.append(f"Attention Entropy (std):   {entropy_stats.get('std', 0):.3f}")
            report.append(f"Fusion Effectiveness:      {metrics.fusion_effectiveness:.3f}")
        
        # Frequency Bin Performance
        if metrics.frequency_bins:
            report.append("\n📊 PERFORMANCE BY FREQUENCY")
            report.append("-" * 40)
            for bin_name, bin_metrics in metrics.frequency_bins.items():
                report.append(f"{bin_name.replace('_', ' ').title()}:")
                report.append(f"  Accuracy: {bin_metrics['accuracy']:.3f}")
                report.append(f"  Avg F1:   {bin_metrics['avg_f1']:.3f}")
                report.append(f"  Classes:  {bin_metrics['num_classes']}")
        
        # Top Performing Classes
        report.append("\n🏆 TOP PERFORMING CLASSES")
        report.append("-" * 40)
        sorted_classes = sorted(
            metrics.class_metrics.items(), 
            key=lambda x: x[1]['f1_score'], 
            reverse=True
        )
        
        for i, (class_name, class_metrics) in enumerate(sorted_classes[:5]):
            report.append(f"{i+1}. {class_name}")
            report.append(f"   F1: {class_metrics['f1_score']:.3f}, "
                         f"Precision: {class_metrics['precision']:.3f}, "
                         f"Recall: {class_metrics['recall']:.3f}")
        
        # Worst Performing Classes
        report.append("\n⚠️  WORST PERFORMING CLASSES")
        report.append("-" * 40)
        for i, (class_name, class_metrics) in enumerate(sorted_classes[-5:]):
            report.append(f"{i+1}. {class_name}")
            report.append(f"   F1: {class_metrics['f1_score']:.3f}, "
                         f"Precision: {class_metrics['precision']:.3f}, "
                         f"Recall: {class_metrics['recall']:.3f}")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)
    
    def save_detailed_results(self, filepath: str, metrics: LongTailMetrics):
        """Save detailed evaluation results to JSON."""
        results = {
            'overall_metrics': {
                'overall_accuracy': metrics.overall_accuracy,
                'macro_f1': metrics.macro_f1,
                'weighted_f1': metrics.weighted_f1,
                'head_accuracy': metrics.head_accuracy,
                'tail_accuracy': metrics.tail_accuracy,
                'few_shot_accuracy': metrics.few_shot_accuracy,
                'zero_shot_accuracy': metrics.zero_shot_accuracy
            },
            'class_metrics': metrics.class_metrics,
            'retrieval_metrics': {
                'precision_at_k': metrics.retrieval_precision_at_k,
                'discriminative_score_correlation': metrics.discriminative_score_correlation
            },
            'fusion_metrics': {
                'attention_entropy_stats': metrics.attention_entropy_stats,
                'fusion_effectiveness': metrics.fusion_effectiveness
            },
            'frequency_bin_metrics': metrics.frequency_bins,
            'evaluation_metadata': {
                'num_predictions': len(self.predictions),
                'num_classes': len(set(self.true_labels + self.predictions)),
                'class_distribution': dict(Counter(self.true_labels))
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Detailed results saved to {filepath}")
