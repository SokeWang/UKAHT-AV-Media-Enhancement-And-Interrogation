# Baseline & Academic Research Plan: UKAHT Multimodal Image Search & Alignment System
[中文版本 (Chinese Version)](./baseline_plan_zh.md)

This baseline plan is designed specifically for a Data Science graduation thesis, focusing on **Multimodal Representation Learning** and **Domain Adaptation**. The system rejects traditional discrete object-label pipelines and instead adopts an end-to-end joint semantic vector space alignment, utilizing a trainable projection adapter to optimize polar domain search and recommendations.

---

## 1. Technical Baseline Designs

### A. End-to-End Multimodal Representation Learning (CLIP & BLIP Baseline)
* **Objective:** Construct a mapping pipeline for images and natural language text within a shared semantic space (Shared Latent Space).
* **Baseline Algorithm:**
  - **Contrastive Representation Learning:** Utilize a pre-trained **CLIP** model (`openai/clip-vit-base-patch32`) as a dual-tower encoder for text and images. Extract 512-dimensional dense embeddings to compute cosine similarity directly.
  - **Image Captioning:** Use a pre-trained **BLIP** model (`Salesforce/blip-image-captioning-base`) to automatically generate descriptive natural language text captions for image assets, establishing a robust textual metadata foundation.

### B. Domain Adaptation MLP Adapter (Data Science Core Contribution)
* **Objective:** Address "Domain Shift" where general pre-trained CLIP models perform suboptimally on specialized polar archives, historic wood buildings, and decay artefacts.
* **Algorithm Design:**
  - Append a trainable 2-layer Multi-Layer Perceptron (MLP) Projection Adapter ($f_\theta(x)$) on top of the frozen 512-dimensional CLIP embedding layers.
  - **Loss Function Optimization:** Train the projection layer using **Triplet Loss / Contrastive Loss** formulated as:
    $$\mathcal{L} = \max(0, d(a, p) - d(a, n) + \text{margin})$$
    This pulls polar images and corresponding oral history transcript quotes (Anchor-Positive pairs) closer in the representation space while pushing irrelevant matches (Anchor-Negative pairs) further apart.

### C. Multimodal Weighted Late Fusion Retrieval
* **Objective:** Combine direct visual patterns with generated text descriptions to perform unified hybrid semantic recall.
* **Algorithm Design:**
  - Retrieve the image visual embedding alongside the text embedding of its generated BLIP caption.
  - Perform similarity computations using Weighted Late Fusion:
    $$S_{final} = \alpha \cdot S_{visual} + (1 - \alpha) \cdot S_{textual}$$
    where $\alpha$ is a balancing factor regulating visual vs. textual relevance weights.

### D. Search Evaluation (Data Science Metrics)
* **Objective:** Quantitatively evaluate the retrieval and recommendation accuracy using statistical rank-based metrics.
* **Evaluation Design:**
  - Curate a custom polar "Golden Test Set" containing specific search tasks and human relevance annotations.
  - Evaluate and compare "Raw CLIP search", "Adapter-aligned search", and "Multimodal Fusion search" using **MAP (Mean Average Precision)** and **nDCG (Normalized Discounted Cumulative Gain)** metrics.

---

## 2. Dependencies to Add (`backend/requirements.txt`)
```text
# Multimodal Foundations & Deep Learning
transformers>=4.30.0
torch>=2.0.0 --index-url https://download.pytorch.org/ml/cpu

# Vector Computations & Visualizations
numpy>=1.24.0
pillow>=9.5.0
scikit-learn>=1.2.0
matplotlib>=3.7.0
```

---

## 3. Verification & Scientific Experiments

### Automated Verify Script
- Write a baseline test script in `backend/verify_baselines.py` to verify the execution of feature extraction, image-to-image similarity scoring, and feature fusion pipeline operations.

### Data Science Research Experiments (Jupyter Notebook)
- Develop a dedicated training and analysis Notebook (e.g., `train_adapter.ipynb`) to conduct:
  1. **Loss Convergence Profiling**: Track and plot the Triplet Loss curve across training epochs.
  2. **Feature Space Visualization**: Apply **t-SNE / UMAP** algorithms to reduce the 512-dimensional embeddings to 2D scatter plots, demonstrating structural clustering differences before and after adapter training.
  3. **Rank Metric Sensitivity Studies**: Plot Precision-Recall curves and evaluate MAP/nDCG sensitivities to changing weights $\alpha$.
