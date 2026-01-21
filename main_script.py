"""
Main script for training and evaluating GBERT-Lite
"""

import torch
import yaml
import argparse
import numpy as np
from pathlib import Path
from transformers import DistilBertTokenizer, GPT2Tokenizer
import sys
import random

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from data.data_loader import DataLoaderFactory
from models.gbert_lite import GBERTLite
from models.baselines import get_baseline_model
from training.trainer import Trainer
from evaluation.evaluator import Evaluator
from evaluation.benchmark import BenchmarkRunner


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main(args):
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load tokenizers
    print("Loading tokenizers...")
    bert_tokenizer = DistilBertTokenizer.from_pretrained(
        config['model']['distilbert_model']
    )
    gpt_tokenizer = GPT2Tokenizer.from_pretrained(
        config['model']['distilgpt2_model']
    )
    gpt_tokenizer.pad_token = gpt_tokenizer.eos_token
    
    if args.mode == 'train':
        train_model(config, bert_tokenizer, gpt_tokenizer, device, args)
    elif args.mode == 'evaluate':
        evaluate_model(config, bert_tokenizer, gpt_tokenizer, device, args)
    elif args.mode == 'benchmark':
        run_benchmark(config, bert_tokenizer, gpt_tokenizer, device, args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


def train_model(config, bert_tokenizer, gpt_tokenizer, device, args):
    """Train GBERT-Lite or baseline model"""
    
    # Run multiple seeds if specified
    num_seeds = config['training'].get('num_seeds', 1)
    all_results = []
    
    for seed_idx in range(num_seeds):
        seed = 42 + seed_idx
        print(f"\n{'='*60}")
        print(f"Training with seed {seed} ({seed_idx + 1}/{num_seeds})")
        print(f"{'='*60}\n")
        
        # Set seed
        set_seed(seed)
        
        # Create data loaders
        print("Creating data loaders...")
        data_factory = DataLoaderFactory(config, bert_tokenizer, gpt_tokenizer)
        dataloaders = data_factory.create_dataloaders(seed=seed)
        
        # Create model
        print(f"Creating model: {args.model}...")
        if args.model == 'gbert_lite':
            model = GBERTLite(config)
        else:
            model = get_baseline_model(args.model, config)
        
        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Create trainer
        trainer = Trainer(model, config, device=device)
        
        # Train
        num_epochs = config['training']['num_epochs']
        history = trainer.train(
            dataloaders['train'],
            dataloaders['val'],
            num_epochs
        )
        
        # Save checkpoint
        checkpoint_dir = Path(config['paths']['checkpoint_dir'])
        checkpoint_path = checkpoint_dir / f"{args.model}_seed{seed}.pt"
        trainer.save_checkpoint(str(checkpoint_path))
        
        # Evaluate on test set
        print("\nEvaluating on test set...")
        evaluator = Evaluator(model, device=device)
        test_metrics = evaluator.evaluate(dataloaders['test'])
        
        print("\nTest Results:")
        for key, value in test_metrics.items():
            print(f"{key}: {value:.4f}")
        
        # Measure latency
        latency_stats = evaluator.measure_latency(
            dataloaders['test'],
            num_samples=config['evaluation']['latency_samples']
        )
        
        print("\nLatency Statistics:")
        for key, value in latency_stats.items():
            print(f"{key}: {value:.2f}")
        
        # Generate report
        results_dir = Path(config['paths']['results_dir']) / f"{args.model}_seed{seed}"
        evaluator.generate_report(str(results_dir))
        
        all_results.append({
            'seed': seed,
            'metrics': test_metrics,
            'latency': latency_stats
        })
    
    # Compute statistics across seeds
    if num_seeds > 1:
        print(f"\n{'='*60}")
        print("Statistics across all seeds:")
        print(f"{'='*60}\n")
        
        metric_names = ['accuracy', 'precision', 'recall', 'f1']
        for metric in metric_names:
            values = [r['metrics'][metric] for r in all_results]
            mean_val = np.mean(values)
            std_val = np.std(values)
            print(f"{metric}: {mean_val:.2f} ± {std_val:.2f}")


def evaluate_model(config, bert_tokenizer, gpt_tokenizer, device, args):
    """Evaluate a trained model"""
    
    # Set seed
    set_seed(42)
    
    # Create data loaders
    print("Creating data loaders...")
    data_factory = DataLoaderFactory(config, bert_tokenizer, gpt_tokenizer)
    dataloaders = data_factory.create_dataloaders(seed=42)
    
    # Create model
    print(f"Creating model: {args.model}...")
    if args.model == 'gbert_lite':
        model = GBERTLite(config)
    else:
        model = get_baseline_model(args.model, config)
    
    # Load checkpoint
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model = model.to(device)
    
    # Evaluate
    evaluator = Evaluator(model, device=device)
    
    print("Evaluating...")
    test_metrics = evaluator.evaluate(dataloaders['test'])
    
    print("\nTest Results:")
    for key, value in test_metrics.items():
        print(f"{key}: {value:.4f}")
    
    # Measure latency
    latency_stats = evaluator.measure_latency(
        dataloaders['test'],
        num_samples=config['evaluation']['latency_samples']
    )
    
    print("\nLatency Statistics:")
    for key, value in latency_stats.items():
        print(f"{key}: {value:.2f}")
    
    # Generate report
    results_dir = Path(config['paths']['results_dir']) / f"{args.model}_evaluation"
    evaluator.generate_report(str(results_dir))


def run_benchmark(config, bert_tokenizer, gpt_tokenizer, device, args):
    """Run comprehensive benchmark across all models"""
    
    print("Running comprehensive benchmark...")
    
    # Create data loaders
    data_factory = DataLoaderFactory(config, bert_tokenizer, gpt_tokenizer)
    dataloaders = data_factory.create_dataloaders(seed=42)
    
    # Run benchmark
    benchmark = BenchmarkRunner(config, device)
    results = benchmark.run_full_benchmark(
        dataloaders,
        bert_tokenizer,
        gpt_tokenizer
    )
    
    # Generate comparison plots
    plots_dir = Path(config['paths']['plots_dir'])
    benchmark.plot_comparison(results, str(plots_dir))
    
    print(f"\nBenchmark results saved to {plots_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GBERT-Lite Training and Evaluation")
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'evaluate', 'benchmark'],
        default='train',
        help='Mode: train, evaluate, or benchmark'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='gbert_lite',
        choices=['gbert_lite', 'bert', 'roberta', 'deberta', 'gbert', 'bert_gnn'],
        help='Model to train/evaluate'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to config file'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint for evaluation'
    )
    
    args = parser.parse_args()
    
    main(args)