"""
algorithm/main.py
Owner: Antigravity AI Assistant — Algorithm FastAPI application entry point

Responsibilities:
  - Define all HTTP API endpoints for deep learning model inference (CLIP, BLIP, Adapter, LangChain Agent)
  - Expose endpoints to the Backend FastAPI service
  - Run with:
      uvicorn algorithm.main:app --port 8001 --reload
"""

import os
from typing import Optional, List
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from algorithm.models.blip_model import get_caption, get_captions_batch
from algorithm.models.clip_model import get_image_embedding, get_image_embeddings_batch, get_text_embedding
from algorithm.models.adapter import load_adapter, apply_adapter
from algorithm.agent.react_agent import ReActAgent

app = FastAPI(
    title="UKAHT Multimodal Algorithm Service",
    version="2.0.0",
    description="Inference endpoints for CLIP, BLIP, polar projection adapter, and LLM agent."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Startup Model Preloading (Warm up Ollama)
# ---------------------------------------------------------------------------
@app.on_event("startup")
def startup_event():
    import threading
    import requests
    import time

    def preload_model():
        model_name = os.getenv("UKAHT_LLM_MODEL", "gemma4:e4b")
        base_url = os.getenv("UKAHT_LLM_BASE_URL", "http://ollama:11434/v1")
        native_url = base_url.replace("/v1", "")
        
        # Wait a few seconds for Ollama container to start up and be responsive
        time.sleep(5)
        
        try:
            print(f"[INFO] Preloading Ollama model {model_name} in background...")
            payload = {"model": model_name}
            resp = requests.post(f"{native_url}/api/generate", json=payload, timeout=180)
            if resp.status_code == 200:
                print(f"[INFO] Successfully preloaded model {model_name} into VRAM.")
            else:
                print(f"[WARN] Failed to preload model {model_name}: {resp.text}")
        except Exception as exc:
            print(f"[WARN] Failed to preload model {model_name}: {exc}")

    threading.Thread(target=preload_model, daemon=True).start()


# ---------------------------------------------------------------------------
# Lazy adapter loader
# ---------------------------------------------------------------------------
_adapter = None
_adapter_loaded = False
STATIC_DIR = os.getenv("UKAHT_STATIC_DIR", "/app/backend/static")

def _get_adapter(adapter_path: Optional[str] = None):
    global _adapter, _adapter_loaded
    if not adapter_path:
        adapter_path = os.path.join(STATIC_DIR, "models", "adapter.pth")
        
    # If a specific path is requested or not loaded yet
    if not _adapter_loaded or (adapter_path and not os.path.exists(os.path.join(STATIC_DIR, "models", "adapter.pth"))):
        _adapter = load_adapter(adapter_path)  # returns None if not yet trained
        _adapter_loaded = True
    return _adapter

# ---------------------------------------------------------------------------
# Stateful LLM Agent Store
# ---------------------------------------------------------------------------
_agent_sessions: dict = {}

def _get_agent(session_id: str):
    if session_id not in _agent_sessions:
        _agent_sessions[session_id] = ReActAgent()
    return _agent_sessions[session_id]

# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------
class TextRequest(BaseModel):
    text: str

class AdaptRequest(BaseModel):
    embeddings: List[List[float]]
    adapter_path: Optional[str] = None

class BatchPathsRequest(BaseModel):
    paths: List[str]

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/algo/health")
def health_check():
    """Liveness check."""
    return {"code": 200, "message": "success", "data": {"status": "ok", "version": app.version}}


@app.post("/api/algo/caption")
async def api_caption(path: Optional[str] = None, file: Optional[UploadFile] = File(None)):
    """Generate caption for a single image, via file path or raw upload."""
    try:
        if file is not None:
            content = await file.read()
            caption = get_caption(content)
        elif path:
            caption = get_caption(path)
        else:
            raise HTTPException(status_code=400, detail="Either 'path' parameter or uploaded 'file' is required")
        return {"code": 200, "message": "success", "data": {"caption": caption}}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/algo/caption/batch")
def api_caption_batch(req: BatchPathsRequest):
    """Generate captions for a batch of local image paths."""
    try:
        captions = get_captions_batch(req.paths)
        return {"code": 200, "message": "success", "data": {"captions": captions}}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/algo/embed/image")
async def api_embed_image(path: Optional[str] = None, file: Optional[UploadFile] = File(None)):
    """Generate CLIP embedding for a single image, via file path or raw upload."""
    try:
        if file is not None:
            content = await file.read()
            emb = get_image_embedding(content)
        elif path:
            emb = get_image_embedding(path)
        else:
            raise HTTPException(status_code=400, detail="Either 'path' parameter or uploaded 'file' is required")
        return {"code": 200, "message": "success", "data": {"embedding": emb.tolist()}}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/algo/embed/image/batch")
def api_embed_image_batch(req: BatchPathsRequest):
    """Generate CLIP embeddings for a batch of local image paths."""
    try:
        embs = get_image_embeddings_batch(req.paths)
        embs_list = [emb.tolist() for emb in embs]
        return {"code": 200, "message": "success", "data": {"embeddings": embs_list}}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/algo/embed/text")
def api_embed_text(req: TextRequest):
    """Generate CLIP embedding for a text query."""
    try:
        emb = get_text_embedding(req.text)
        return {"code": 200, "message": "success", "data": {"embedding": emb.tolist()}}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/algo/adapt")
def api_adapt(req: AdaptRequest):
    """Project CLIP embeddings using the MLP Adapter."""
    try:
        import numpy as np
        adapter = _get_adapter(req.adapter_path)
        
        # If adapter is not trained yet, return original embeddings
        if adapter is None:
            return {"code": 200, "message": "success", "data": {"embeddings": req.embeddings}}
            
        embeddings = np.array(req.embeddings, dtype=np.float32)
        adapted = apply_adapter(embeddings, adapter)
        return {"code": 200, "message": "success", "data": {"embeddings": adapted.tolist()}}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/algo/agent/chat")
def api_agent_chat(req: ChatRequest):
    """Run one turn of the LangChain LLM ReAct agent."""
    try:
        session_id = req.session_id or "default"
        agent = _get_agent(session_id)
        result = agent.run(req.message)
        return {
            "code": 200,
            "message": "success",
            "data": {
                "answer": result["answer"],
                "retrieved_assets": result["retrieved_assets"],
                "session_id": session_id,
            }
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.delete("/api/algo/agent/session/{session_id}")
def api_reset_session(session_id: str):
    """Reset LLM agent conversation history for a given session."""
    if session_id in _agent_sessions:
        _agent_sessions[session_id].reset()
    return {"code": 200, "message": "success", "data": {"status": "reset", "session_id": session_id}}
