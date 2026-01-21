"""
Comprehensive evaluation module
Metrics, latency analysis, and visualization
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
import json


class Evaluator:
    """Comprehensive evaluator for fake news detection models"""
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model.to(device)
        self.device = device
        self.results = {}
    
    def evaluate(self, test_loader: DataLoader, 
                threshold: float = 0.5) -> Dict[str, float]:
        """
        Comprehensive evaluation on test set
        
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        print("Evaluating model...")
        
        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                outputs = self.model(batch)
                probs = outputs['falsity_prob'].squeeze().cpu().numpy()
                labels = batch['labels'].cpu().numpy()
                
                all_probs.extend(probs if isinstance(probs, np.ndarray) else [probs])
                all_labels.extend(labels if isinstance(labels, np.ndarray) else [labels])
        
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        all_preds = (all_probs >= threshold).astype(int)
        
        # Compute metrics
        metrics = {
            'accuracy': accuracy_score(all_labels, all_preds) * 100,
            'precision': precision_score(all_labels, all_preds, zero_division=0) * 100,
            'recall': recall_score(all_labels, all_preds, zero_division=0) * 100,
            'f1': f1_score(all_labels, all_preds, zero_division=0) * 100,
            'roc_auc': roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.0,
            'pr_auc': average_precision_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.0
        }
        
        # Store for later use
        self.results['predictions'] = all_preds
        self.results['probabilities'] = all_probs
        self.results['labels'] = all_labels
        self.results['metrics'] = metrics
        
        return metrics
    
    def measure_latency(self, test_loader: DataLoader, 
                       num_samples: int = 500) -> Dict[str, float]:
        """
        Measure inference latency
        
        Args:
            test_loader: Test data loader (should have batch_size=1)
            num_samples: Number of samples to measure
        
        Returns:
            Latency statistics
        """
        self.model.eval()
        
        latencies = []
        count = 0
        
        print(f"Measuring latency on {num_samples} samples...")
        
        with torch.no_grad():
            for batch in test_loader:
                if count >= num_samples:
                    break
                
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                # Warm-up GPU
                if count == 0:
                    for _ in range(10):
                        _ = self.model(batch)
                
                # Measure
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start_time = time.time()
                
                _ = self.model(batch)
                
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.time()
                
                latencies.append((end_time - start_time) * 1000)  # Convert to ms
                count += 1
        
        latencies = np.array(latencies)
        
        latency_stats = {
            'mean_latency_ms': np.mean(latencies),
            'std_latency_ms': np.std(latencies),
            'min_latency_ms': np.min(latencies),
            'max_latency_ms': np.max(latencies),
            'median_latency_ms': np.median(latencies)
        }
        
        self.results['latency'] = latency_stats
        
        return latency_stats
    
    def compute_confusion_matrix(self) -> np.ndarray:
        """Compute confusion matrix"""
        if 'predictions' not in self.results:
            raise ValueError("Run evaluate() first")
        
        cm = confusion_matrix(
            self.results['labels'],
            self.results['predictions']
        )
        
        self.results['confusion_matrix'] = cm
        return cm
    
    def plot_confusion_matrix(self, save_path: str = None):
        """Plot confusion matrix"""
        if 'confusion_matrix' not in self.results:
            self.compute_confusion_matrix()
        
        cm = self.results['confusion_matrix']
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Real News', 'Fake News'],
                   yticklabels=['Real News', 'Fake News'])
        plt.title('Confusion Matrix: GBERT-Lite')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_roc_curve(self, save_path: str = None):
        """Plot ROC curve"""
        if 'probabilities' not in self.results:
            raise ValueError("Run evaluate() first")
        
        fpr, tpr, thresholds = roc_curve(
            self.results['labels'],
            self.results['probabilities']
        )
        
        auc = self.results['metrics']['roc_auc']
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'GBERT-Lite (AUC={auc:.3f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend()
        plt.grid(alpha=0.3)
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_precision_recall_curve(self, save_path: str = None):
        """Plot Precision-Recall curve"""
        if 'probabilities' not in self.results:
            raise ValueError("Run evaluate() first")
        
        precision, recall, thresholds = precision_recall_curve(
            self.results['labels'],
            self.results['probabilities']
        )
        
        ap = self.results['metrics']['pr_auc']
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'GBERT-Lite (AP={ap:.3f})', linewidth=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(alpha=0.3)
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_error_distribution(self, save_path: str = None):
        """Plot error distribution (FP vs FN)"""
        if 'confusion_matrix' not in self.results:
            self.compute_confusion_matrix()
        
        cm = self.results['confusion_matrix']
        fp = cm[0, 1]  # False Positives
        fn = cm[1, 0]  # False Negatives
        
        plt.figure(figsize=(8, 6))
        plt.bar(['False Positives', 'False Negatives'], [fp, fn],
               color=['#ff6b6b', '#4ecdc4'])
        plt.ylabel('Error Count')
        plt.title('Error Distribution: GBERT-Lite')
        
        for i, v in enumerate([fp, fn]):
            plt.text(i, v + 5, str(v), ha='center', fontweight='bold')
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def save_results(self, save_path: str):
        """Save evaluation results to JSON"""
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare serializable results
        save_dict = {
            'metrics': self.results.get('metrics', {}),
            'latency': self.results.get('latency', {}),
            'confusion_matrix': self.results.get('confusion_matrix', np.array([])).tolist()
        }
        
        with open(save_path, 'w') as f:
            json.dump(save_dict, f, indent=2)
        
        print(f"Results saved to {save_path}")
    
    def generate_report(self, save_dir: str):
        """Generate comprehensive evaluation report"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print("Generating evaluation report...")
        
        # Save plots
        self.plot_confusion_matrix(save_dir / "confusion_matrix.png")
        self.plot_roc_curve(save_dir / "roc_curve.png")
        self.plot_precision_recall_curve(save_dir / "pr_curve.png")
        self.plot_error_distribution(save_dir / "error_distribution.png")
        
        # Save results
        self.save_results(save_dir / "results.json")
        
        # Generate text report
        report = self._generate_text_report()
        with open(save_dir / "report.txt", 'w') as f:
            f.write(report)
        
        print(f"Report generated in {save_dir}")
    
    def _generate_text_report(self) -> str:
        """Generate text report"""
        metrics = self.results.get('metrics', {})
        latency = self.results.get('latency', {})
        
        report = "=" * 50 + "\n"
        report += "GBERT-Lite Evaluation Report\n"
        report += "=" * 50 + "\n\n"
        
        report += "Performance Metrics:\n"
        report += "-" * 50 + "\n"
        for key, value in metrics.items():
            report += f"{key.capitalize():20s}: {value:.2f}"
            if key not in ['roc_auc', 'pr_auc']:
                report += "%"
            report += "\n"
        
        report += "\n" + "Latency Statistics:\n"
        report += "-" * 50 + "\n"
        for key, value in latency.items():
            report += f"{key:20s}: {value:.2f} ms\n"
        
        if 'confusion_matrix' in self.results:
            cm = self.results['confusion_matrix']
            report += "\n" + "Confusion Matrix:\n"
            report += "-" * 50 + "\n"
            report += f"True Negatives:  {cm[0, 0]}\n"
            report += f"False Positives: {cm[0, 1]}\n"
            report += f"False Negatives: {cm[1, 0]}\n"
            report += f"True Positives:  {cm[1, 1]}\n"
        
        return report