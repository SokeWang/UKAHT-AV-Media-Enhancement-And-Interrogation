# Antarctic Multimodal Search & Representation Learning System: 14-Week Project Plan (UKAHT)
[中文版本 (Chinese Version)](./project_plan_14_weeks_zh.md)

This document outlines a structured, 14-week project plan aligned with the core Data Science architecture:  
**Antarctic Image $\rightarrow$ Multimodal Representation Models (CLIP & BLIP) $\rightarrow$ [Domain Adaptation MLP Adapter, Weighted Late Fusion, Search Evaluation (MAP/nDCG)]**

This project adopts a **Dual-Track Delivery Strategy** containing **Academic Research Notebooks** (focusing on Triplet Loss optimization and t-SNE clustering visualizations) and a **Prototype Application** (implemented with FastAPI and React).

---

### **Decoupled Architecture (Inference Interfaces)**
To support independent algorithm fine-tuning and parameter updates, the system uses strict function boundaries:
1. `get_image_embedding(image_source) -> np.ndarray` (extract 512-dim visual embeddings)
2. `get_text_embedding(text_query) -> np.ndarray` (extract 512-dim query text embeddings)
3. `get_blip_captions_batch(images) -> List[str]` (generate image text descriptions)
4. `classify_images_batch(images) -> List[dict]` (generate zero-shot image class labels)

---

### **14-Week Schedule**

#### **Phase 1: MVP Prototyping & Database Design (Weeks 1-2)**
*Focus: Establish a fully working mock frontend-backend skeleton, then design SQLite storage schemas.*
- **Week 1: Retrieval System MVP Skeleton**
  - Setup React + Vite frontend and FastAPI backend. Configure proxy settings to enable clean cross-origin requests.
  - Implement mock/dummy versions of core vector search functions to establish search grids and recommended detail sliders.
- **Week 2: Data Audit & SQLite Vector Database Schema**
  - Audit the UKAHT Antarctic multimedia library, removing corrupt files and cataloging folders.
  - Setup the SQLite schema to store metadata alongside the raw 512-dimensional embeddings as BLOB arrays.
  - Establish API and path compatibility conventions for cross-platform execution.

#### **Phase 2: Baseline Model Integration & Batch Indexing (Weeks 3-5)**
*Focus: Replace mock APIs with pre-trained CLIP and BLIP models, then run batch ingestion pipeline.*
- **Week 3: Multimodal Extraction Baseline Integration**
  - Replace mock files with pre-trained CLIP (`openai/clip-vit-base-patch32`) for image/text embeddings and BLIP (`Salesforce/blip-image-captioning-base`) for text captions.
- **Week 4: Image Library Batch Indexing (Ingestion)**
  - Implement batch model inference pipelines (Batch Size = 4) and SQLite batch transaction commits to optimize ingestion speed.
  - Index the entire polar image collection, computing visual categories, textual captions, and vector embeddings.
- **Week 5: Semantic Retrieval System Stabilization**
  - Fine-tune backend endpoints for text search, visual recommendations, and oral transcript keyword filters to guarantee baseline stability.

#### **Phase 3: Domain Adaptation MLP Adapter Development (Weeks 6-8)**
*Focus: Build and train a custom projection adapter using Triplet Loss to correct CLIP domain shift.*
- **Week 6: Triplet Dataset Construction**
  - Pair polar images with corresponding oral history transcripts to construct positive pairs.
  - Sample unrelated texts and images to construct negative samples.
  - Set up the training data generator outputting Triplet batches (Anchor, Positive, Negative).
- **Week 7: MLP Adapter Training**
  - Implement a 2-layer MLP projection network in PyTorch, defining a custom Triplet Loss with margin.
  - Train the projection weights on the custom polar dataset to map generic representations into polar-aligned vector representations.
- **Week 8: Adaptive Alignment Integration**
  - Save the trained adapter weights (`.pth`) and load them inside the feature extraction pipeline, confirming seamless integration and backward compatibility.

#### **Phase 4: Multimodal Fusion & Rank-based Quantitative Evaluation (Weeks 9-10)**
*Focus: Construct hybrid similarity search methods and evaluate results using information retrieval metrics.*
- **Week 9: Multimodal Late Fusion Development**
  - Implement similarity weighting algorithms combining image visual embeddings with generated description text embeddings: $S_{final} = \alpha S_{visual} + (1-\alpha)S_{textual}$.
- **Week 10: Golden Test Set Benchmarking (Ablation Analysis)**
  - Annotate a test collection of specific search tasks with human relevance scores.
  - Evaluate and compare search configurations (Raw CLIP, Adapter-aligned, and Multimodal Fusion) on **MAP (Mean Average Precision)** and **nDCG (Normalized Discounted Cumulative Gain)** metrics, plotting Precision-Recall curves.

#### **Phase 5: Academic Visualizations & Hand-off Prep (Weeks 11-13)**
*Focus: Conduct dimensionality reduction analysis for feature clustering and prepare hand-off scripts.*
- **Week 11: Feature Space Clustering & t-SNE Plotting**
  - Write evaluation Notebooks applying **t-SNE / UMAP** algorithms to reduce 512-dim spaces to 2D scatter plots, visualising clustering separation before and after adapter alignment.
- **Week 12: Application Polishing**
  - Refine frontend visual indicators and search interfaces for oral transcripts and recommendations.
- **Week 13: Delivery Packaging (Dockerization)**
  - Write `Dockerfile` and `docker-compose.yml` configurations, packaging the application in a CPU-only Debian container to ensure universal deployment.

#### **Phase 6: Project Handoff (Week 14)**
*Focus: Final deliverables.*
- **Week 14: System Acceptance Testing & Documentation Handoff**
  - Compile final experimental accuracy tables, write the research methodology chapter for the thesis, and hand over the codebase repository.
