# Antarctic Multimodal Search & Representation Learning System: 6-Week Project Plan (UKAHT)
[中文版本 (Chinese Version)](./project_plan_14_weeks_zh.md)

This document outlines a structured, 6-week project plan aligned with the core Data Science architecture:  
**Antarctic Image $\rightarrow$ Multimodal Representation Models (CLIP & BLIP) $\rightarrow$ [Domain Adaptation MLP Adapter, Weighted Late Fusion, Search Evaluation (MAP/nDCG)]**

To address timeline and feasibility risks, the project adopts a **"Baseline-First" approach** (deploying a working pre-trained CLIP/BLIP pipeline first) before layering on polar domain fine-tuning and LLM RAG agents. Each milestone's explicit deliverables are linked directly to team roles.

---

### **Team Organization & Responsibilities**

* **Peidong Wang**
  * **DS Core**: Domain Adaptation & Multimodal Representation (design and implement custom MLP Adapter layer, formulate Triplet Loss, correct CLIP polar domain drift).
  * **Engineering**: Backend Infrastructure & API Logic (develop asynchronous RESTful APIs using FastAPI, encapsulate model inference, provide stable RAG endpoints).
* **Tian Luo**
  * **DS Core**: LLM Agent & Intelligent RAG Architecture (engineer ReAct prompt strategies, implement tool-calling and intent-parsing).
  * **Engineering**: Hybrid Database & Retrieval Engine (design SQLite vector + relational schema, implement integrated queries, optimize database indexes).
* **Yisheng Zhang**
  * **DS Core**: Automated Processing & Feature Engineering (deploy BLIP for batch captioning, run 3.3 GB ingestion pipeline, extract CLIP visual/text embeddings).
  * **Engineering**: DevOps & Containerization (write Dockerfiles, configure docker-compose orchestration, manage dependencies).
* **Chenyu Yuan**
  * **DS Core**: Academic Evaluation & Visualization (program evaluation scripts for MAP and nDCG, run ablation studies, build t-SNE/UMAP plots).
  * **Engineering**: Interactive Streamlit UI (build chat panel, result grids, recommendation cards, and human-in-the-loop annotation UI).

---

### **6-Week Schedule**

#### **Phase 1: Core Baseline Pipeline & Verification (Weeks 1-2)**
*Focus: Establish a fully working mock and baseline pipeline to guarantee feasibility early on.*
* **Explicit Outputs**:
  1. **SQLite Seed Database**: SQLite file `backend/db.sqlite` with `assets` table schema configured for metadata, text descriptions, and 512-dim vector embeddings (BLOBs), populated with initial mock assets.
  2. **FastAPI Services**: Working `/api/search` (semantic search) and `/api/recommend` (visual similarity recommendation) API routes.
  3. **Frontend Search Grid**: Streamlit/React search panel communicating with backend APIs.
  4. **Baseline Verification Script**: A passing `backend/verify_baselines.py` script outputting similarity scores, validating end-to-end connectivity.
* **Team Mapping**:
  * **Peidong Wang**: Implement pre-trained CLIP/BLIP inference APIs and set up FastAPI structure.
  * **Tian Luo**: Design and initialize the SQLite storage schema.
  * **Chenyu Yuan**: Design the base Streamlit/React search and grid UI.

#### **Phase 2: Corpus Ingestion & Golden Test Set Curation (Week 3)**
*Focus: Ingest the full 3.3 GB polar image database and establish the validation benchmark.*
* **Explicit Outputs**:
  1. **Corpus Indexed Database**: SQLite database populated with auto-generated BLIP captions and CLIP visual embeddings for all 3.3 GB of image files.
  2. **Golden Test Set File**: A `golden_test_set.json` file containing 50–100 manually curated "strong positive" image-query benchmark pairs.
  3. **Annotation Interface**: Interactive human-in-the-loop annotation UI panel on the frontend to review and refine captions.
* **Team Mapping**:
  * **Yisheng Zhang**: Implement batch BLIP captioning pipeline and run 3.3 GB dataset ingestion into SQLite.
  * **Tian Luo**: Optimize SQLite batch ingestion transactional logic to speed up database commits.
  * **Chenyu Yuan**: Build the Streamlit assisted annotation page, collaborating with the team to compile the JSON benchmark.

#### **Phase 3: Domain Adaptation MLP Adapter (Week 4)**
*Focus: Train the MLP projection adapter using Triplet Loss to correct CLIP polar domain drift.*
* **Explicit Outputs**:
  1. **Triplet Training Pipeline**: A PyTorch-compatible data generator outputting (Anchor, Positive, Negative) samples.
  2. **Trained Model Weights**: A saved PyTorch adapter weights file `adapter.pth`.
  3. **Loss Analysis Notebook**: `train_adapter.ipynb` containing loss curves plotting Triplet Loss convergence.
* **Team Mapping**:
  * **Peidong Wang**: Design the 2-layer MLP projection adapter, implement the Triplet Loss function, train the model, and integrate it into backend feature extraction.
  * **Yisheng Zhang**: Write the random sampling script to generate positive and negative pairs for training.

#### **Phase 4: LLM Agent & RAG Pipeline Integration (Week 5)**
*Focus: Implement the ReAct agent framework to support multi-turn conversational searches.*
* **Explicit Outputs**:
  1. **Query Intent Router**: ReAct-style prompt templates and routing scripts converting natural language queries into combined Vector + SQL parameters.
  2. **RAG Context Synthesis**: Multi-turn text generation pipelines merging visual search results with conversation histories using an LLM.
  3. **Conversational Interface**: A sliding drawer chat interface integrated on the frontend.
* **Team Mapping**:
  * **Tian Luo**: Design the agent prompt strategies and tool-calling hybrid query execution logic.
  * **Peidong Wang**: Deploy RAG FastAPI endpoints and link the adapter-aligned visual query extraction.
  * **Chenyu Yuan**: Build and styled the multi-turn chat panel in the UI.

#### **Phase 5: Evaluation, High-D Visualization, & Packaging (Week 6)**
*Focus: Run statistical evaluation of retrieval precision, visualize high-dim clusters, and package for handoff.*
* **Explicit Outputs**:
  1. **Evaluation Tables**: A comparison table in the evaluation Notebook showing **MAP** and **nDCG** scores across configurations (Raw CLIP vs. Adapter-Aligned vs. Late Fusion).
  2. **Clustering Scatters**: 2D scatter plots generated via t-SNE/UMAP in the Notebook, visualizing cluster boundary separation before and after adapter alignment.
  3. **Docker Configurations**: A clean, CPU-compatible `Dockerfile` and `docker-compose.yml` configuration launching frontend and backend services.
* **Team Mapping**:
  * **Chenyu Yuan**: Write evaluation scripts for MAP/nDCG metrics and generate t-SNE/UMAP scatter plots.
  * **Yisheng Zhang**: Write Dockerfiles and configure docker-compose containers for cross-platform deployment.
