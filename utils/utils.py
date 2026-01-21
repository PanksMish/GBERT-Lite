"""
Utility functions for GBERT-Lite
"""

import torch
import numpy as np
import random
import json
import yaml
from pathlib import Path
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import seaborn as sns


def set_seed(seed: int):
    """
    Set random seed for reproducibility
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to config file
    
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], save_path: str):
    """
    Save configuration to YAML file
    
    Args:
        config: Configuration dictionary
        save_path: Path to save config
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def save_json(data: Dict, save_path: str):
    """
    Save dictionary to JSON file
    
    Args:
        data: Dictionary to save
        save_path: Path to save JSON
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(json_path: str) -> Dict:
    """
    Load dictionary from JSON file
    
    Args:
        json_path: Path to JSON file
    
    Returns:
        Loaded dictionary
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Count model parameters
    
    Args:
        model: PyTorch model
    
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': frozen_params
    }


def format_number(num: int) -> str:
    """
    Format large numbers with commas
    
    Args:
        num: Number to format
    
    Returns:
        Formatted string
    """
    return f"{num:,}"


def get_gpu_memory() -> Dict[str, float]:
    """
    Get GPU memory usage
    
    Returns:
        Dictionary with memory stats (in GB)
    """
    if not torch.cuda.is_available():
        return {'allocated': 0, 'reserved': 0, 'total': 0}
    
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    return {
        'allocated': allocated,
        'reserved': reserved,
        'total': total,
        'free': total - allocated
    }


def plot_training_history(history: Dict[str, List[float]], save_path: str = None):
    """
    Plot training history
    
    Args:
        history: Dictionary with train/val losses
        save_path: Path to save plot
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    epochs = range(1, len(history['train_losses']) + 1)
    ax.plot(epochs, history['train_losses'], 'b-', label='Training Loss')
    ax.plot(epochs, history['val_losses'], 'r-', label='Validation Loss')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training History')
    ax.legend()
    ax.grid(alpha=0.3)
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def create_directory_structure():
    """Create necessary directories for the project"""
    directories = [
        'data',
        'checkpoints',
        'results',
        'plots',
        'logs',
        'cache'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("Directory structure created successfully!")


def print_model_summary(model: torch.nn.Module, model_name: str = "Model"):
    """
    Print model summary
    
    Args:
        model: PyTorch model
        model_name: Name of the model
    """
    param_counts = count_parameters(model)
    
    print(f"\n{'='*60}")
    print(f"{model_name} Summary")
    print(f"{'='*60}")
    print(f"Total Parameters:     {format_number(param_counts['total'])}")
    print(f"Trainable Parameters: {format_number(param_counts['trainable'])}")
    print(f"Frozen Parameters:    {format_number(param_counts['frozen'])}")
    
    if torch.cuda.is_available():
        memory = get_gpu_memory()
        print(f"\nGPU Memory:")
        print(f"  Allocated: {memory['allocated']:.2f} GB")
        print(f"  Free:      {memory['free']:.2f} GB")
        print(f"  Total:     {memory['total']:.2f} GB")
    
    print(f"{'='*60}\n")


def aggregate_results(results_list: List[Dict]) -> Dict:
    """
    Aggregate results from multiple runs
    
    Args:
        results_list: List of result dictionaries
    
    Returns:
        Aggregated statistics
    """
    if not results_list:
        return {}
    
    # Get all metric names
    metric_names = list(results_list[0]['metrics'].keys())
    
    aggregated = {}
    for metric in metric_names:
        values = [r['metrics'][metric] for r in results_list]
        aggregated[f'{metric}_mean'] = np.mean(values)
        aggregated[f'{metric}_std'] = np.std(values)
        aggregated[f'{metric}_min'] = np.min(values)
        aggregated[f'{metric}_max'] = np.max(values)
    
    return aggregated


def print_results_table(results: Dict[str, Dict]):
    """
    Print results in a formatted table
    
    Args:
        results: Dictionary of model results
    """
    print(f"\n{'='*80}")
    print("BENCHMARK RESULTS")
    print(f"{'='*80}")
    
    # Header
    print(f"{'Model':<15} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Latency (ms)':<12}")
    print(f"{'-'*80}")
    
    # Results
    for model_name, model_results in results.items():
        acc = f"{model_results['accuracy_mean']:.2f}±{model_results['accuracy_std']:.2f}"
        prec = f"{model_results['precision_mean']:.2f}±{model_results['precision_std']:.2f}"
        rec = f"{model_results['recall_mean']:.2f}±{model_results['recall_std']:.2f}"
        f1 = f"{model_results['f1_mean']:.2f}±{model_results['f1_std']:.2f}"
        lat = f"{model_results['latency_mean']:.2f}"
        
        print(f"{model_name.upper():<15} {acc:<12} {prec:<12} {rec:<12} {f1:<12} {lat:<12}")
    
    print(f"{'='*80}\n")


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.001):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, val_score: float) -> bool:
        """
        Check if training should stop
        
        Args:
            val_score: Validation score (higher is better)
        
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = val_score
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.counter = 0
        
        return self.early_stop


def compute_class_weights(labels: np.ndarray) -> torch.Tensor:
    """
    Compute class weights for imbalanced datasets
    
    Args:
        labels: Array of labels
    
    Returns:
        Class weights tensor
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(labels)
    weights = compute_class_weight('balanced', classes=classes, y=labels)
    
    return torch.FloatTensor(weights)
