Below is a **clean, professional `README.md`** suitable for **GitHub upload**, aligned with **SCI-quality research code**, and clearly reflecting the scope of your paper and experiments.

You can copy–paste this directly into a file named **`README.md`**.

---

````markdown
# GBERT-Lite

**GBERT-Lite** is a lightweight dual-transformer framework for **diffusion-aware and risk-sensitive misinformation detection**.  
The model integrates frozen **DistilBERT** and **DistilGPT-2** encoders to capture complementary contextual and generative semantics, while maintaining low computational and memory overhead suitable for real-world deployment.

This repository accompanies the research paper:

> **GBERT-Lite: A Lightweight Dual-Transformer Framework for Diffusion-Aware Misinformation Detection**  
> *Submitted to Computers & Electrical Engineering (Elsevier)*

---

## 📌 Key Features

- Dual-encoder fusion using frozen DistilBERT and DistilGPT-2  
- Lightweight classification head trained under low-VRAM constraints  
- Diffusion-aware risk modelling for decision-oriented mitigation  
- High predictive performance with reduced inference latency  
- Extensive evaluation including comparative study and ablation analysis  

---

## 🏗️ Architecture Overview

The GBERT-Lite framework consists of three main stages:

1. **Semantic Encoding**  
   - DistilBERT for bidirectional contextual semantics  
   - DistilGPT-2 for autoregressive and narrative cues  

2. **Diffusion-Aware Risk Estimation**  
   - Probabilistic falsity prediction  
   - Temporal propagation scoring  

3. **Policy-Oriented Decision Module**  
   - Threshold-based risk-aware mitigation  

---

## 📊 Experimental Results (Summary)

| Metric | Value |
|------|------|
| Accuracy | 98.40% |
| Precision | 99.24% |
| Recall | 97.24% |
| F1-score | 98.23% |
| AUC | 0.983 |
| PR-AUC | 0.978 |

GBERT-Lite outperforms strong transformer and hybrid baselines while achieving over **50% lower inference latency** on a **4 GB GPU**.

---

## 🧪 Evaluation and Visualizations

The repository includes scripts to generate:

- Metric comparison plots  
- ROC and Precision–Recall curves  
- Confusion matrices and error analysis  
- Inference latency benchmarks  
- Ablation study visualizations  
- Performance–efficiency trade-off analysis  

All plots are IEEE/Springer publication-ready.

---

## 🗂️ Repository Structure

```text
.
├── data/                  # Dataset preprocessing scripts / placeholders
├── models/                # Model definitions and checkpoints
├── experiments/           # Training and evaluation scripts
├── plots/                 # Generated figures (PDF/PNG)
├── utils/                 # Helper functions and utilities
├── README.md              # Project documentation
└── requirements.txt       # Python dependencies
````

---

## ⚙️ Requirements

* Python ≥ 3.9
* PyTorch ≥ 2.0
* Transformers (HuggingFace) ≥ 4.35
* Scikit-learn
* NumPy, Matplotlib

Install dependencies using:

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage (High-Level)

1. Preprocess the dataset
2. Run training with frozen encoders
3. Evaluate on the test set
4. Generate plots and ablation results

Detailed scripts and instructions are provided in the `experiments/` directory.

---

## 👥 Authors

* **Pankaj Mishra** (Corresponding Author)
  K. J. Somaiya School of Engineering, India

* **Sneha Nagarajan**
  K. J. Somaiya School of Engineering, India

* **V. Venkataramanan**
  K. J. Somaiya School of Engineering, India

* **Weiwei Jiang**
  Beijing University of Posts and Telecommunications, China

---

## 📄 License

This project is provided for **research and academic use only**.
Please cite the associated paper if you use this code in your work.

---

## 📬 Contact

For questions, feedback, or collaboration inquiries:

📧 **[pankaj.mishra@somaiya.edu](mailto:pankaj.mishra@somaiya.edu)**

---

## 📚 Citation

Citation details will be added upon publication.

```

---

If you want, I can also:
- Add a **BibTeX citation block**
- Create a **clean `requirements.txt`**
- Prepare a **code-of-conduct / license file**
- Make the README **conference-specific or journal-linked**

Just say the word 👍
```
