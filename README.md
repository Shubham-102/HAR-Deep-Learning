# 🏃 Human Activity Recognition — Capstone Project

> **Multi-Modal Deep Learning · Explainability · Robustness Analysis**  
> M.Eng. Computer Science Capstone · University of Connecticut · May 2026

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![SHAP](https://img.shields.io/badge/Explainability-SHAP-FF6F61?style=flat)](https://shap.readthedocs.io/)
[![Colab](https://img.shields.io/badge/Open_in-Colab-F9AB00?style=flat&logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/Shubham-102/Capstone-Project/blob/main/Capstone.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E?style=flat)](LICENSE)

---

## 📌 Overview

This project implements and compares **7 deep learning architectures** for Human Activity Recognition (HAR) on the UCI HAR benchmark dataset — from a simple MLP baseline to a gated multi-modal fusion network combining raw sensor signals with handcrafted features.

The pipeline covers the full ML lifecycle:

- ✅ **EDA** on 10,299 samples across 6 activity classes
- ✅ **Subject-based 70/30 split** to prevent data leakage and ensure genuine generalisation
- ✅ **7 neural network architectures** across 3 tiers of complexity
- ✅ **Robustness testing** under Gaussian noise, feature dropout, and combined degradation
- ✅ **SHAP Gradient Explainer** for global and per-class feature importance
- ✅ **Self-supervised pre-training** (masked signal reconstruction)
- ✅ **Knowledge distillation** — TinyHAR student model (7× smaller than teacher)
- ✅ **Cross-dataset generalisation** tested on WISDM dataset
- ✅ **Fall detection case study** for elderly care applications

---

## 🏆 Results at a Glance

| Model | Category | Test Accuracy | Macro F1 | Robustness Drop |
|---|---|:---:|:---:|:---:|
| SimpleMLP | Feature-only | 94.10% | 0.941 | −17.4 pp |
| RegMLP | Feature-only | 94.98% | 0.950 | −16.7 pp |
| DeepMLP | Feature-only | 95.49% | 0.955 | −18.7 pp |
| **FusionNet ★** | **Multi-modal** | **96.64%** | **0.966** | **−16.4 pp** |
| GatedFusionNet | Multi-modal | 95.45% | 0.954 | −19.9 pp |
| BiLSTM | Advanced | — | — | — |
| CNN-LSTM | Advanced | — | — | — |

> ★ **FusionNet** achieves the best clean-data accuracy **and** the smallest robustness drop, making it the recommended model for real-world deployment.

---

## 🧠 Model Architectures

```
Tier 1 — Feature-Only (MLP Family)
├── SimpleMLP       — 2-layer baseline, no regularisation
├── RegMLP          — + Dropout (p=0.35)
└── DeepMLP         — + BatchNorm, deeper funnel architecture

Tier 2 — Multi-Modal Fusion
├── FusionNet       — Parallel CNN (raw signals) + MLP (features), concatenated
└── GatedFusionNet  — Learnable gate α dynamically weights each modality per sample

Tier 3 — Advanced Sequential
├── BiLSTM          — Bidirectional LSTM for long-range temporal dependencies
└── CNN-LSTM        — CNN local feature extraction → LSTM temporal modelling
```

**Multi-modal input shape:** `(N, 3, 128)` raw tri-axial signals + `(N, 561)` handcrafted features  
**Shared TSEncoder:** 3-layer Conv1D → AdaptiveAvgPool → fixed-size embedding

---

## 📊 Dataset — UCI HAR

| Property | Detail |
|---|---|
| Source | Anguita et al., ESANN 2013 |
| Subjects | 30 volunteers, aged 19–48 |
| Sensor | Samsung Galaxy S II (waist-mounted) |
| Sampling rate | 50 Hz |
| Window | 2.56s, 128 timesteps, 50% overlap |
| Features | 561 handcrafted (time + frequency domain) |
| Classes | Walking, Walking Upstairs, Walking Downstairs, Sitting, Standing, Laying |
| Total samples | 10,299 |

**Split strategy:** Subject-based 70/30 (21 train subjects / 9 test subjects). No subject appears in both sets — this tests true generalisation to unseen people, unlike the random splits common in the literature.

---

## 🔬 Key Findings

### 1. Handcrafted features are extremely powerful
Even the bare baseline (SimpleMLP, no regularisation) hits **94.10%** — the 561 domain-expert features encode most of the discriminative signal. Complex architectures improve at the margin.

### 2. Multi-modal fusion's real advantage is robustness
FusionNet's time-series CNN branch acts as a **backup modality** when handcrafted features are corrupted, explaining its smallest robustness drop (−16.4 pp) despite all models being classified as fragile under severe degradation.

### 3. Gated fusion requires degradation-aware training
GatedFusionNet's learned gate α is architecturally superior but was trained on clean data. Under sensor degradation, the gate continues to down-weight the time-series branch at precisely the moments it would be most useful — resulting in the **largest robustness drop (−19.9 pp)**.

### 4. SHAP reveals modality-specific feature subsets
- **Laying** → body orientation features (distinctive gravity vector)
- **Walking variants** → FFT coefficients and spectral entropy (rhythmic gait)
- **Sitting / Standing** → subtle torso inclination features (inherently noisy)

The **top 30–50 features carry the majority of predictive power**, suggesting that 500+ features could be dropped in a resource-constrained deployment with minimal accuracy cost.

---

## 🛡️ Robustness Testing

Three degradation conditions at increasing severity levels:

| Condition | Description |
|---|---|
| Gaussian noise | Additive noise N(0, σ²) on all 561 features |
| Feature zeroing | Random subset of features set to 0 (simulates sensor failure) |
| Combined | Both applied simultaneously |

FusionNet is the only architecture with a structural fallback (raw signal CNN branch) when features are corrupted.

---

## ⚡ Modern Techniques

### Self-Supervised Pre-Training (Masked Signal Reconstruction)
- Masks 30% of raw signal timesteps
- Pre-trains an SSLEncoder to reconstruct masked regions
- Fine-tuned encoder transferred to classification head
- Demonstrates label-efficient learning for wearable scenarios

### Knowledge Distillation — TinyHAR
- **Teacher:** FusionNet (~150K params)
- **Student:** TinyHAR (~8K params, **7× smaller**)
- Trained with KL-divergence soft-target loss (T=4.0, α=0.7)
- Designed to fit on wearable MCUs with 256KB RAM

---

## 🌍 Cross-Dataset Generalisation

FusionNet trained on UCI HAR was evaluated on **WISDM** (pocket-mounted, free-living conditions) to test domain shift robustness. This reflects the real-world gap between lab-controlled collection (UCI HAR) and natural user behaviour (WISDM).

---

## 📚 Literature Comparison

| Reference | Method | Accuracy |
|---|---|:---:|
| Anguita et al. (2013) | SVM | 89.3% |
| Ordóñez & Roggen (2016) | DeepConvLSTM | 91.2% |
| Ullah et al. (2019) | BiLSTM + Attention | 93.7% |
| **This work — FusionNet** | **Multi-modal CNN + MLP** | **96.64%** |

Trends 2023–2025 addressed in this project: Foundation Models for HAR · Knowledge Distillation for TinyML · Cross-Dataset Transfer · Multimodal Sensor Fusion

---

## 🗂️ Project Structure

```
Capstone-Project/
├── Capstone.ipynb          # Complete pipeline (64 cells, 17 sections)
└── README.md
```

**Notebook sections:**
1. Setup & Reproducibility
2. Load Dataset
3. Exploratory Data Analysis
4. Preprocessing — Subject-Based 70/30 Split
5. Model Definitions (7 Architectures)
6. Training Infrastructure
7. Train All 7 Models
8. Evaluation — Accuracy, F1, Confusion Matrices, ROC Curves
9. Robustness Analysis
10. SHAP Explainability
11. Self-Supervised Pre-Training
12. Knowledge Distillation — TinyHAR
13. Case Study — Elderly Fall Detection
14. Real-World Generalisation Study (WISDM)
15. Literature Comparison & Current Trends
16. Save Best Model
17. Final Summary

---

## 🚀 Getting Started

### Run in Google Colab (recommended)

Click the badge at the top or go to:  
**`https://colab.research.google.com/github/Shubham-102/Capstone-Project/blob/main/Capstone.ipynb`**

The notebook auto-downloads the UCI HAR dataset and all dependencies.

### Run Locally

```bash
# Clone the repo
git clone https://github.com/Shubham-102/Capstone-Project.git
cd Capstone-Project

# Install dependencies
pip install torch numpy pandas scikit-learn matplotlib seaborn shap scipy pillow

# Launch notebook
jupyter notebook Capstone.ipynb
```

> ⚠️ **GPU recommended.** Tested on NVIDIA RTX 3070 (8GB VRAM). Training all 7 models takes ~15–20 min on GPU, ~45–60 min on CPU.

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Deep Learning | PyTorch 2.x |
| Data & Features | NumPy, Pandas, SciPy, Scikit-learn |
| Explainability | SHAP (GradientExplainer) |
| Visualization | Matplotlib, Seaborn |
| Environment | Google Colab / Jupyter |

---

## 📖 References

- Anguita, D., et al. (2013). *A Public Domain Dataset for Human Activity Recognition Using Smartphones.* ESANN 2013.
- Lundberg, S. M., & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions.* NeurIPS 2017.
- Paszke, A., et al. (2019). *PyTorch: An Imperative Style, High-Performance Deep Learning Library.* NeurIPS 2019.
- Ioffe, S., & Szegedy, C. (2015). *Batch Normalization: Accelerating Deep Network Training.* ICML 2015.

---

## 👤 Author

**Shubham Maheshwari**  
M.Eng. Computer Science · University of Connecticut (May 2026)  
[LinkedIn](https://linkedin.com/in/maheshwari-shubham) · [Portfolio](https://shubham-102.github.io) · [GitHub](https://github.com/Shubham-102)

---

*Supervised by Prof. Chuxu Zhang · Department of Computer Science & Engineering · UConn*
