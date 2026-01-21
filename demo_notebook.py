{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# GBERT-Lite Demo Notebook\n",
    "\n",
    "This notebook demonstrates how to use GBERT-Lite for fake news detection."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Setup and Imports"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch\n",
    "import yaml\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "from transformers import DistilBertTokenizer, GPT2Tokenizer\n",
    "from pathlib import Path\n",
    "\n",
    "# Import GBERT-Lite components\n",
    "from models.gbert_lite import GBERTLite\n",
    "from data.data_loader import DataLoaderFactory, TextPreprocessor\n",
    "from training.trainer import Trainer\n",
    "from evaluation.evaluator import Evaluator\n",
    "from utils.utils import set_seed, load_config\n",
    "\n",
    "# Set device\n",
    "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
    "print(f\"Using device: {device}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Load Configuration"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load config\n",
    "config = load_config('config.yaml')\n",
    "\n",
    "# Set seed for reproducibility\n",
    "set_seed(42)\n",
    "\n",
    "print(\"Configuration loaded successfully!\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Load Tokenizers"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load tokenizers\n",
    "bert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')\n",
    "gpt_tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')\n",
    "gpt_tokenizer.pad_token = gpt_tokenizer.eos_token\n",
    "\n",
    "print(\"Tokenizers loaded!\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Load Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create data loaders\n",
    "data_factory = DataLoaderFactory(config, bert_tokenizer, gpt_tokenizer)\n",
    "dataloaders = data_factory.create_dataloaders(seed=42)\n",
    "\n",
    "print(f\"Train samples: {len(dataloaders['train'].dataset)}\")\n",
    "print(f\"Val samples: {len(dataloaders['val'].dataset)}\")\n",
    "print(f\"Test samples: {len(dataloaders['test'].dataset)}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Create Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create GBERT-Lite model\n",
    "model = GBERTLite(config)\n",
    "model = model.to(device)\n",
    "\n",
    "# Print model summary\n",
    "total_params = model.get_total_parameters()\n",
    "trainable_params = model.get_trainable_parameters()\n",
    "\n",
    "print(f\"Total parameters: {total_params:,}\")\n",
    "print(f\"Trainable parameters: {trainable_params:,}\")\n",
    "print(f\"Frozen parameters: {total_params - trainable_params:,}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Train Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create trainer\n",
    "trainer = Trainer(model, config, device=device)\n",
    "\n",
    "# Train\n",
    "history = trainer.train(\n",
    "    dataloaders['train'],\n",
    "    dataloaders['val'],\n",
    "    num_epochs=3\n",
    ")\n",
    "\n",
    "print(\"Training completed!\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 7. Evaluate Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create evaluator\n",
    "evaluator = Evaluator(model, device=device)\n",
    "\n",
    "# Evaluate\n",
    "test_metrics = evaluator.evaluate(dataloaders['test'])\n",
    "\n",
    "print(\"\\nTest Results:\")\n",
    "for metric, value in test_metrics.items():\n",
    "    print(f\"{metric}: {value:.4f}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 8. Measure Latency"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Measure inference latency\n",
    "latency_stats = evaluator.measure_latency(dataloaders['test'], num_samples=100)\n",
    "\n",
    "print(\"\\nLatency Statistics:\")\n",
    "for metric, value in latency_stats.items():\n",
    "    print(f\"{metric}: {value:.2f} ms\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 9. Generate Visualizations"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Generate comprehensive report\n",
    "evaluator.generate_report('./results/demo')\n",
    "\n",
    "print(\"Report generated in ./results/demo/\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 10. Make Predictions on Custom Text"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def predict_news(text, model, bert_tokenizer, gpt_tokenizer, device):\n",
    "    \"\"\"\n",
    "    Predict if news is fake or real\n",
    "    \"\"\"\n",
    "    model.eval()\n",
    "    \n",
    "    # Preprocess\n",
    "    preprocessor = TextPreprocessor(config['preprocessing'])\n",
    "    text = preprocessor.clean_text(text)\n",
    "    \n",
    "    # Tokenize\n",
    "    bert_encoding = bert_tokenizer(\n",
    "        text,\n",
    "        max_length=64,\n",
    "        padding='max_length',\n",
    "        truncation=True,\n",
    "        return_tensors='pt'\n",
    "    )\n",
    "    \n",
    "    gpt_encoding = gpt_tokenizer(\n",
    "        text,\n",
    "        max_length=64,\n",
    "        padding='max_length',\n",
    "        truncation=True,\n",
    "        return_tensors='pt'\n",
    "    )\n",
    "    \n",
    "    # Create batch\n",
    "    batch = {\n",
    "        'bert_input_ids': bert_encoding['input_ids'].to(device),\n",
    "        'bert_attention_mask': bert_encoding['attention_mask'].to(device),\n",
    "        'gpt_input_ids': gpt_encoding['input_ids'].to(device),\n",
    "        'gpt_attention_mask': gpt_encoding['attention_mask'].to(device)\n",
    "    }\n",
    "    \n",
    "    # Predict\n",
    "    with torch.no_grad():\n",
    "        outputs = model(batch)\n",
    "        prob = outputs['falsity_prob'].item()\n",
    "    \n",
    "    prediction = \"FAKE\" if prob >= 0.5 else \"REAL\"\n",
    "    confidence = prob if prob >= 0.5 else 1 - prob\n",
    "    \n",
    "    return prediction, confidence\n",
    "\n",
    "# Test on custom text\n",
    "sample_text = \"Breaking: Scientists discover cure for all diseases!\"\n",
    "prediction, confidence = predict_news(sample_text, model, bert_tokenizer, gpt_tokenizer, device)\n",
    "\n",
    "print(f\"\\nText: {sample_text}\")\n",
    "print(f\"Prediction: {prediction}\")\n",
    "print(f\"Confidence: {confidence:.2%}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 11. Save Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Save checkpoint\n",
    "trainer.save_checkpoint('./checkpoints/gbert_lite_demo.pt')\n",
    "print(\"Model saved!\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Conclusion\n",
    "\n",
    "This notebook demonstrated:\n",
    "1. Loading and preprocessing data\n",
    "2. Creating and training GBERT-Lite\n",
    "3. Evaluating performance\n",
    "4. Measuring inference latency\n",
    "5. Making predictions on custom text\n",
    "\n",
    "For more information, see the README.md file."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}