"""
Benchmark runner for comparing multiple models
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

from models.gbert_lite import GBERTLite
from models.baselines import get_baseline_model
from training.trainer import Trainer
from evaluation.evaluator import Evaluator


class BenchmarkRunner:
    """Run comprehensive benchmarks across multiple models"""
    
    def __init__(self, config: Dict, device: str = 'cuda'):
        self.config = config
        self.device = device
        self.models = ['bert', 'roberta', 'deberta', 'gbert', 'bert_gnn', 'gbert_lite']
    
    def run_full_benchmark(self, dataloaders: Dict, 
                          bert_tokenizer, gpt_tokenizer) -> Dict:
        """Run full benchmark on all models"""
        
        results = {}
        
        for model_name in self.models:
            print(f"\n{'='*60}")
            print(f"Benchmarking: {model_name.upper()}")
            print(f"{'='*60}\n")
            
            # Run multiple seeds
            num_seeds = self.config['training'].get('num_seeds', 5)
            seed_results = []
            
            for seed_idx in range(num_seeds):
                seed = 42 + seed_idx
                print(f"Seed {seed} ({seed_idx + 1}/{num_seeds})")
                
                # Set seed
                torch.manual_seed(seed)
                np.random.seed(seed)
                
                # Create model
                if model_name == 'gbert_lite':
                    model = GBERTLite(self.config)
                else:
                    model = get_baseline_model(model_name, self.config)
                
                # Train
                trainer = Trainer(model, self.config, device=self.device)
                trainer.train(
                    dataloaders['train'],
                    dataloaders['val'],
                    self.config['training']['num_epochs']
                )
                
                # Evaluate
                evaluator = Evaluator(model, device=self.device)
                metrics = evaluator.evaluate(dataloaders['test'])
                latency = evaluator.measure_latency(
                    dataloaders['test'],
                    num_samples=500
                )
                
                seed_results.append({
                    'metrics': metrics,
                    'latency': latency['mean_latency_ms']
                })
            
            # Aggregate results
            aggregated = self._aggregate_results(seed_results)
            results[model_name] = aggregated
            
            print(f"\n{model_name.upper()} Results:")
            print(f"Accuracy: {aggregated['accuracy_mean']:.2f} ± {aggregated['accuracy_std']:.2f}")
            print(f"F1: {aggregated['f1_mean']:.2f} ± {aggregated['f1_std']:.2f}")
            print(f"Latency: {aggregated['latency_mean']:.2f} ms")
        
        return results
    
    def _aggregate_results(self, seed_results: List[Dict]) -> Dict:
        """Aggregate results across seeds"""
        
        metrics_names = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']
        aggregated = {}
        
        for metric in metrics_names:
            values = [r['metrics'][metric] for r in seed_results]
            aggregated[f'{metric}_mean'] = np.mean(values)
            aggregated[f'{metric}_std'] = np.std(values)
        
        latencies = [r['latency'] for r in seed_results]
        aggregated['latency_mean'] = np.mean(latencies)
        aggregated['latency_std'] = np.std(latencies)
        
        return aggregated
    
    def plot_comparison(self, results: Dict, save_dir: str):
        """Generate comparison plots"""
        
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot 1: Metric comparison
        self._plot_metrics_comparison(results, save_dir / "metrics_comparison.png")
        
        # Plot 2: Performance trends
        self._plot_performance_trends(results, save_dir / "performance_trends.png")
        
        # Plot 3: Latency comparison
        self._plot_latency_comparison(results, save_dir / "latency_comparison.png")
        
        # Plot 4: Performance-efficiency tradeoff
        self._plot_efficiency_tradeoff(results, save_dir / "efficiency_tradeoff.png")
        
        print(f"Plots saved to {save_dir}")
    
    def _plot_metrics_comparison(self, results: Dict, save_path: Path):
        """Plot metrics comparison across models"""
        
        models = list(results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        
        x = np.arange(len(models))
        width = 0.2
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for i, metric in enumerate(metrics):
            values = [results[m][f'{metric}_mean'] for m in models]
            ax.bar(x + i * width, values, width, label=metric.capitalize())
        
        ax.set_xlabel('Model Architecture')
        ax.set_ylabel('Performance Score (%)')
        ax.set_title('Performance Comparison Across Model Architectures')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels([m.upper() for m in models], rotation=45)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_trends(self, results: Dict, save_path: Path):
        """Plot performance trends"""
        
        models = list(results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for metric in metrics:
            values = [results[m][f'{metric}_mean'] for m in models]
            ax.plot(models, values, marker='o', linewidth=2, label=metric.capitalize())
        
        ax.set_xlabel('Model Architecture')
        ax.set_ylabel('Performance Metric (%)')
        ax.set_title('Performance Metric Trends Across Models')
        ax.set_xticklabels([m.upper() for m in models], rotation=45)
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_latency_comparison(self, results: Dict, save_path: Path):
        """Plot latency comparison"""
        
        models = list(results.keys())
        latencies = [results[m]['latency_mean'] for m in models]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(models, latencies, color='steelblue')
        
        # Annotate bars
        for bar, latency in zip(bars, latencies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{latency:.1f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_xlabel('Model Architecture')
        ax.set_ylabel('Inference Latency (ms/sample)')
        ax.set_title('Inference Time Comparison')
        ax.set_xticklabels([m.upper() for m in models], rotation=45)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_efficiency_tradeoff(self, results: Dict, save_path: Path):
        """Plot performance-efficiency tradeoff"""
        
        models = list(results.keys())
        f1_scores = [results[m]['f1_mean'] for m in models]
        latencies = [results[m]['latency_mean'] for m in models]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for i, model in enumerate(models):
            ax.scatter(latencies[i], f1_scores[i], s=200, alpha=0.7,
                      label=model.upper())
            ax.annotate(model.upper(), (latencies[i], f1_scores[i]),
                       xytext=(5, 5), textcoords='offset points')
        
        ax.set_xlabel('Inference Latency (ms/sample)')
        ax.set_ylabel('F1-Score (%)')
        ax.set_title('Performance-Efficiency Trade-off')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()