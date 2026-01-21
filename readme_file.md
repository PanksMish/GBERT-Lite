# GBERT-Lite: Lightweight Dual-Transformer Framework for Diffusion-Aware Misinformation Detection

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation of **GBERT-Lite: A Lightweight Dual-Transformer Framework for Diffusion-Aware Misinformation Detection**.

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Quick Start](#quick-start)
- [Training](#training)
- [Evaluation](#evaluation)
- [Benchmarking](#benchmarking)
- [Results](#results)
- [Project Structure](#project-structure)
- [Citation](#citation)

## 🔍 Overview

GBERT-Lite is a lightweight dual-transformer architecture that combines frozen DistilBERT and DistilGPT-2 encoders for efficient misinformation detection. The framework integrates:

- **Dual-Encoder Fusion**: Complementary bidirectional contextual and autoregressive representations
- **Diffusion-Aware Risk Modeling**: Temporal propagation scorer for impact assessment
- **Policy-Driven Mitigation**: Game-theoretic decision framework for intervention

### Performance Highlights

- **98.40%** Accuracy
- **99.24%** Precision
- **97.24%** Recall
- **98.23%** F1-Score
- **50%+** Latency Reduction (vs. full transformers)

## ✨ Key Features

- ✅ Frozen transformer encoders (minimal training overhead)
- ✅ Efficient dual-encoder architecture
- ✅ Diffusion-aware risk scoring
- ✅ Comprehensive benchmarking against BERT, RoBERTa, DeBERTa, GBERT, BERT+GNN
- ✅ Resource-constrained deployment (4GB GPU)
- ✅ Multi-seed evaluation with statistical analysis
- ✅ Extensive visualization and reporting

## 🚀 Installation

### Prerequisites

- Python 3.10+
- CUDA 11.x (for GPU support)
- 4GB+ GPU (tested on GTX 1650)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/gbert-lite.git
cd gbert-lite

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model (optional, for preprocessing)
python -m spacy download en_core_web_sm
```

## 📊 Dataset Setup

### Fake and Real News Dataset

1. Download from [Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
2. Place `Fake.csv` and `True.csv` in `./data/`

```bash
# Using Kaggle API
kaggle datasets download -d clmentbisaillon/fake-and-real-news-dataset
unzip fake-and-real-news-dataset.zip -d ./data/
```

### FakeNewsNet (Optional)

1. Clone [FakeNewsNet repository](https://github.com/KaiDMML/FakeNewsNet)
2. Follow their instructions to download data
3. Place processed data in `./data/fakenewsnet/`

## 🏃 Quick Start

### Train GBERT-Lite

```bash
python main.py --mode train --model gbert_lite
```

### Evaluate Trained Model

```bash
python main.py --mode evaluate --model gbert_lite --checkpoint ./checkpoints/gbert_lite_seed42.pt
```

### Run Full Benchmark

```bash
python main.py --mode benchmark
```

## 🎯 Training

### Configuration

Edit `config.yaml` to customize:

```yaml
model:
  freeze_encoders: true
  max_length: 64

training:
  batch_size: 4
  learning_rate: 2.0e-5
  num_epochs: 3
  num_seeds: 5
```

### Train Specific Models

```bash
# GBERT-Lite
python main.py --mode train --model gbert_lite

# BERT Baseline
python main.py --mode train --model bert

# RoBERTa Baseline
python main.py --mode train --model roberta

# DeBERTa Baseline
python main.py --mode train --model deberta

# GBERT Baseline
python main.py --mode train --model gbert

# BERT+GNN Baseline
python main.py --mode train --model bert_gnn
```

### Multi-Seed Training

The framework automatically runs training with multiple seeds (default: 5) and reports mean ± std statistics.

## 📈 Evaluation

### Single Model Evaluation

```bash
python main.py --mode evaluate --model gbert_lite --checkpoint path/to/checkpoint.pt
```

### Generated Outputs

Evaluation generates:

- `confusion_matrix.png` - Confusion matrix visualization
- `roc_curve.png` - ROC curve
- `pr_curve.png` - Precision-Recall curve
- `error_distribution.png` - False positive vs false negative distribution
- `results.json` - Numerical results
- `report.txt` - Comprehensive text report

## 🏆 Benchmarking

### Run Comprehensive Benchmark

```bash
python main.py --mode benchmark
```

This will:

1. Train all models (BERT, RoBERTa, DeBERTa, GBERT, BERT+GNN, GBERT-Lite)
2. Evaluate each with 5 different seeds
3. Measure inference latency
4. Generate comparison plots

### Benchmark Outputs

- `metrics_comparison.png` - Side-by-side metric comparison
- `performance_trends.png` - Performance trends across models
- `latency_comparison.png` - Inference time comparison
- `efficiency_tradeoff.png` - F1 score vs latency scatter plot

## 📊 Results

### Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score | Latency (ms) |
|-------|----------|-----------|--------|----------|--------------|
| BERT | 95.20±0.48 | 94.80±0.52 | 95.60±0.44 | 95.10±0.46 | 42.5 |
| RoBERTa | 96.40±0.41 | 96.70±0.38 | 96.00±0.43 | 96.30±0.40 | 38.2 |
| DeBERTa | 96.80±0.39 | 97.20±0.36 | 96.40±0.41 | 96.80±0.38 | 40.1 |
| GBERT | 95.30±0.45 | 95.10±0.49 | 97.30±0.42 | 96.20±0.44 | 57.2 |
| BERT+GNN | - | - | - | - | 49.3 |
| **GBERT-Lite** | **98.40±0.21** | **99.24±0.18** | **97.24±0.23** | **98.23±0.20** | **18.9** |

### Key Findings

- **Best Performance**: GBERT-Lite achieves highest accuracy, precision, and F1-score
- **Lowest Latency**: 50%+ faster than full transformer baselines
- **Most Stable**: Lowest variance across random seeds
- **Resource Efficient**: Runs on 4GB GPU with frozen encoders

## 📁 Project Structure

```
gbert-lite/
├── config.yaml                 # Configuration file
├── main.py                     # Main training/evaluation script
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── data/                       # Data directory
│   ├── data_loader.py         # Dataset loading and preprocessing
│   ├── Fake.csv               # Fake news dataset (download separately)
│   └── True.csv               # Real news dataset (download separately)
│
├── models/                     # Model implementations
│   ├── gbert_lite.py          # GBERT-Lite architecture
│   └── baselines.py           # Baseline models
│
├── training/                   # Training utilities
│   └── trainer.py             # Trainer class
│
├── evaluation/                 # Evaluation utilities
│   ├── evaluator.py           # Evaluation metrics and visualization
│   └── benchmark.py           # Benchmark runner
│
├── checkpoints/                # Saved model checkpoints
├── results/                    # Evaluation results
├── plots/                      # Generated plots
└── logs/                       # Training logs
```

## 🔧 Advanced Usage

### Custom Configuration

Create a custom config file:

```bash
python main.py --config custom_config.yaml --mode train
```

### Ablation Studies

Modify `config.yaml` to test different configurations:

```yaml
model:
  freeze_encoders: false  # Fine-tune encoders
  classifier_hidden_dim: 512  # Larger classifier

diffusion:
  enabled: false  # Disable diffusion modeling
```

### GPU Memory Optimization

For limited GPU memory:

```yaml
training:
  batch_size: 2
  mixed_precision: true
```

## 🐛 Troubleshooting

### CUDA Out of Memory

- Reduce `batch_size` in `config.yaml`
- Enable `mixed_precision: true`
- Reduce `max_length` from 64 to 32

### Dataset Not Found

Ensure datasets are in correct location:

```
data/
  ├── Fake.csv
  └── True.csv
```

### Slow Training

- Enable mixed precision training
- Use smaller batch sizes with gradient accumulation
- Ensure CUDA is properly installed

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@article{mishra2025gbert,
  title={GBERT-Lite: A Lightweight Dual-Transformer Framework for Diffusion-Aware Misinformation Detection},
  author={Mishra, Pankaj and Nagarajan, Sneha and Venkataramanan, V. and Jiang, Weiwei},
  journal={arXiv preprint},
  year={2025}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

- Pankaj Mishra - pankaj.mishra@somaiya.edu
- Project Link: [https://github.com/yourusername/gbert-lite](https://github.com/yourusername/gbert-lite)

## 🙏 Acknowledgments

- K J Somaiya School of Engineering
- Beijing University of Posts and Telecommunications
- Hugging Face Transformers library
- PyTorch team

---

**Note**: This implementation is for research purposes. For production deployment, additional security measures and monitoring should be implemented.