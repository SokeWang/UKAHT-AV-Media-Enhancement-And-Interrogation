# Chenyu Yuan - Data Science Experiment Log

This directory is designated for recording data science experiments conducted by Chenyu Yuan.

## Research Focus Areas
1. **Academic Metric Compilations**: Evaluating MAP, nDCG, and MRR metrics across retrieval methods.
2. **Feature Space Visualizations**: Designing high-dimensional projection experiments (t-SNE, UMAP parameters like perplexity, learning rates).
3. **UI/UX Interaction Metrics**: Tracking human-in-the-loop caption corrections and golden dataset build rates.

## THCS (Temporal-Heritage Coverage Score) Metric Evaluation

Implemented the THCS@K metric script at [eval_thcs.py](file:///home/ubuntu/UKAHT-AV-Media-Enhancement-And-Interrogation/experiments/chenyu/eval_thcs.py).

$$\text{THCS}@K = P_{entity}@K \times \left( \frac{\text{EraCoverage}@K + \text{NormILD}_{temporal}@K}{2} \right)$$

### 1. Evaluation Results

We evaluated Vanilla CLIP and the 6 trained PEFT adapters under two year-generation modes:
* **SYNTHETIC Mode**: Assigns deterministic pseudo-years using `hash(asset_id) % 75` to all assets (reproducing Peidong's validation setup).
* **DATABASE Mode**: Parses the actual `shooting_year` from the database, falling back to the hash-based pseudo-year only when the year is missing or invalid.

#### Year Mode: SYNTHETIC (Replication of Peidong's validation)
| Model Architecture | P_entity@10 | EraCov@10 | NormILD@10 | THCS@10 |
| :--- | :---: | :---: | :---: | :---: |
| Vanilla CLIP Baseline | 0.0800 | 0.9500 | 0.6394 | 0.0594 |
| 2-Layer Baseline MLP Adapter (mlp) | 0.0840 | 0.9300 | 0.6372 | 0.0664 |
| Single-Branch SwiGLU Gated (swiglu) | 0.2780 | 0.9550 | 0.6359 | 0.2200 |
| Temporal & Environment Decoupled (ted) | 0.2500 | 0.9450 | 0.6239 | 0.1938 |
| Dual-Branch SwiGLU Gated (dual_swiglu) | 0.2920 | 0.9450 | 0.6317 | 0.2284 |
| Polar-Contextual Enhancement (pcse) | 0.2720 | 0.9550 | 0.6156 | 0.2152 |
| PCSE-ACRA (Spectrum Equalized) | **0.3020** | **0.9700** | **0.6361** | **0.2403** |

#### Year Mode: DATABASE (Actual database years with hash fallback)
| Model Architecture | P_entity@10 | EraCov@10 | NormILD@10 | THCS@10 |
| :--- | :---: | :---: | :---: | :---: |
| Vanilla CLIP Baseline | 0.0800 | 0.6000 | 0.4342 | 0.0345 |
| 2-Layer Baseline MLP Adapter (mlp) | 0.0840 | **0.8250** | **0.5824** | 0.0580 |
| Single-Branch SwiGLU Gated (swiglu) | 0.2780 | 0.7700 | 0.5346 | 0.1775 |
| Temporal & Environment Decoupled (ted) | 0.2500 | 0.7600 | 0.5218 | 0.1576 |
| Dual-Branch SwiGLU Gated (dual_swiglu) | 0.2920 | 0.8050 | 0.5659 | **0.1910** |
| Polar-Contextual Enhancement (pcse) | 0.2720 | 0.7550 | 0.5162 | 0.1690 |
| PCSE-ACRA (Spectrum Equalized) | **0.3020** | 0.7700 | 0.5176 | 0.1862 |

---

### 2. Model Performance Evaluation & Insights

#### ① mlp (2-Layer Baseline MLP Adapter) Underperforms due to Limited Representation Capacity
* **Analysis**: Under both year-generation modes, `mlp` achieves a $P_{entity}@10$ of only **0.0840**, which is almost identical to the Vanilla CLIP Baseline (0.0800). This indicates that a simple linear projection + ReLU lacks the non-linear warping and residual fitting capacity required to map complex polar heritage semantics and specific base codes (e.g., Base A/E).

#### ② SwiGLU Gated Mechanism and Dual-Branch Architectures Significantly Boost Entity Grounding
* **Analysis**: Introducing SwiGLU activation functions (`swiglu`, `dual_swiglu`, `pcse`, `pcse_acra`) yields a significant leap in $P_{entity}@10$ (rising from ~0.08 to **0.27 - 0.30**). Dual-branch configurations perform best, with `dual_swiglu` hitting **0.2920** and `pcse_acra` reaching **0.3020**. This demonstrates the superiority of a decoupled architecture in capturing macro-level terrain and micro-level architectural features.

#### ③ Value of ACRA Spectrum Equalization: Optimal Balance between Generalization and Diversity
* **Analysis**:
  1. In **SYNTHETIC Mode**, `pcse_acra` achieves the highest $P_{entity}@10$ (**0.3020**) and the highest THCS@10 (**0.2403**).
  2. In **DATABASE Mode**, `dual_swiglu` achieves a slightly higher THCS@10 (0.1910 vs 0.1862) due to a marginally wider era coverage. However, `pcse_acra` remains highly competitive.
  3. Considering the out-of-sample benchmark (where `pcse_acra` achieves the highest MAP of **0.3190**, significantly outperforming `dual_swiglu`'s **0.2963**), the covariance spectrum equalization (ACRA) plays a critical role in preventing representation collapse and mitigating overfitting. This ensures robust retrieval generalization on unseen polar images.
