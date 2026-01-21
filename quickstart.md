# GBERT-Lite Quick Start Guide

Get up and running with GBERT-Lite in 5 minutes!

## Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended, but not required)
- 8GB+ RAM
- 10GB+ free disk space

## Installation Steps

### 1. Clone and Setup

```bash
# Clone repository
git clone https://github.com/yourusername/gbert-lite.git
cd gbert-lite

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Dataset

```bash
# Option 1: Automatic download (requires Kaggle API)
python scripts/download_data.py --dataset fake_real_news

# Option 2: Manual download
# 1. Go to: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
# 2. Download Fake.csv and True.csv
# 3. Place in ./data/ directory
```

### 3. Quick Training

Train GBERT-Lite with default settings:

```bash
python main.py --mode train --model gbert_lite
```

This will:
- Load and preprocess the dataset
- Train GBERT-Lite for 3 epochs
- Evaluate on test set
- Save checkpoint to `./checkpoints/`
- Generate results in `./results/`

### 4. Evaluate Model

```bash
python main.py --mode evaluate --model gbert_lite --checkpoint ./checkpoints/gbert_lite_seed42.pt
```

### 5. Run Full Benchmark

Compare all models:

```bash
python main.py --mode benchmark
```

**Note:** This will take several hours as it trains all baseline models.

## Using Jupyter Notebook

```bash
jupyter notebook demo.ipynb
```

The demo notebook provides an interactive walkthrough.

## Expected Results

After training for 3 epochs, you should see approximately:

- **Accuracy:** ~98%
- **Precision:** ~99%
- **Recall:** ~97%
- **F1-Score:** ~98%
- **Inference Time:** ~18-20ms per sample (on GTX 1650)

## Customization

Edit `config.yaml` to customize:

```yaml
training:
  batch_size: 4        # Reduce if out of memory
  learning_rate: 2.0e-5
  num_epochs: 3        # Increase for better performance
  num_seeds: 5         # Number of random seeds

model:
  freeze_encoders: true  # Keep true for efficiency
  max_length: 64         # Sequence length
```

## Troubleshooting

### Out of Memory Error

```yaml
# In config.yaml
training:
  batch_size: 2          # Reduce batch size
  mixed_precision: true  # Enable FP16
```

### Dataset Not Found

Ensure files are in correct location:
```
data/
  ├── Fake.csv
  └── True.csv
```

### Slow Training

- Enable mixed precision: `mixed_precision: true` in config
- Use GPU: Check `torch.cuda.is_available()`
- Reduce `max_length` from 64 to 32

## Next Steps

1. **Experiment with baselines:**
   ```bash
   python main.py --mode train --model bert
   python main.py --mode train --model roberta
   ```

2. **Try different configurations:**
   - Unfreeze encoders: `freeze_encoders: false`
   - Increase epochs: `num_epochs: 5`
   - Adjust learning rate

3. **Evaluate on custom data:**
   - See `demo.ipynb` for prediction examples
   - Modify data loader for custom datasets

4. **Generate visualizations:**
   - All plots saved in `./plots/`
   - Confusion matrices, ROC curves, etc.

## Common Commands

```bash
# Train GBERT-Lite
python main.py --mode train --model gbert_lite

# Train baseline
python main.py --mode train --model bert

# Evaluate
python main.py --mode evaluate --model gbert_lite --checkpoint path/to/checkpoint.pt

# Full benchmark
python main.py --mode benchmark

# Run tests
pytest tests/

# Download data
python scripts/download_data.py
```

## File Outputs

After running experiments, you'll find:

```
checkpoints/
  └── gbert_lite_seed42.pt    # Trained model

results/
  └── gbert_lite_seed42/
      ├── confusion_matrix.png
      ├── roc_curve.png
      ├── pr_curve.png
      ├── error_distribution.png
      ├── results.json
      └── report.txt

plots/
  ├── metrics_comparison.png
  ├── performance_trends.png
  ├── latency_comparison.png
  └── efficiency_tradeoff.png
```

## Performance Tips

1. **Use GPU:** Ensure CUDA is available
2. **Enable mixed precision:** Set `mixed_precision: true`
3. **Freeze encoders:** Keep `freeze_encoders: true` for speed
4. **Batch size:** Balance between 2-8 based on GPU memory
5. **Cache data:** Processed data is cached in `./cache/`

## Getting Help

- Check [README.md](README.md) for detailed documentation
- Review [demo.ipynb](demo.ipynb) for examples
- Run tests: `pytest tests/`
- Open an issue on GitHub

## Citation

If you use this code, please cite:

```bibtex
@article{mishra2025gbert,
  title={GBERT-Lite: A Lightweight Dual-Transformer Framework for Diffusion-Aware Misinformation Detection},
  author={Mishra, Pankaj and Nagarajan, Sneha and Venkataramanan, V. and Jiang, Weiwei},
  year={2025}
}
```

---

**Happy experimenting! 🚀**