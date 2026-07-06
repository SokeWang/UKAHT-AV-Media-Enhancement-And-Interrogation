"""
backend/main.py
Owner: Peidong Wang — FastAPI application entry point

Responsibilities:
  - Define all HTTP API endpoints (thin routing layer only)
  - Delegate business logic to dedicated modules:
      Search / Recommend  → retrieval/search.py     (Tian Luo)
      Upload + captioning → models/blip_model.py    (Yisheng Zhang)
                            models/clip_model.py    (Peidong Wang)
      Agent Q&A           → agent/react_agent.py    (Tian Luo)
  - Serve static files (uploaded images, dataset)
  - No ML or DB logic lives here

Run with:
  uvicorn backend.main:app --reload
"""

import os
import shutil
import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional

# ---------------------------------------------------------------------------
# Application setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="UKAHT Antarctic Media Interrogation System",
    version="2.0.0",
    description=(
        "Multimodal retrieval and Q&A system for the UK Antarctic Heritage Trust "
        "historical image archive."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Static directories
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
UPLOAD_DIR = os.path.join(STATIC_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

DATASET_DIR = os.getenv("UKAHT_DATA_DIR", "/Users/wangpeidong/UKAHT-Project/UK Antarctic Heritage Trust Data")
if os.path.exists(DATASET_DIR):
    app.mount("/data", StaticFiles(directory=DATASET_DIR), name="data")

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    query: str
    category: Optional[str] = None  # optional category filter


class RecommendRequest(BaseModel):
    id: str
    limit: Optional[int] = 6


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None  # future: per-session history


# ---------------------------------------------------------------------------
# In-memory agent session store  (Milestone 4 — Tian Luo)
# ---------------------------------------------------------------------------

_agent_sessions: dict = {}


def _get_agent(session_id: str):
    from backend.agent.react_agent import ReActAgent
    if session_id not in _agent_sessions:
        _agent_sessions[session_id] = ReActAgent()
    return _agent_sessions[session_id]


# ---------------------------------------------------------------------------
# Lazy adapter loader  (Milestone 3 — Peidong Wang)
# ---------------------------------------------------------------------------

_adapter = None
_adapter_loaded = False


def _get_adapter():
    global _adapter, _adapter_loaded
    if not _adapter_loaded:
        from backend.models.adapter import load_adapter
        adapter_path = os.path.join(STATIC_DIR, "models", "adapter.pth")
        _adapter = load_adapter(adapter_path)  # returns None if not yet trained
        _adapter_loaded = True
    return _adapter


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health_check():
    """Quick liveness check."""
    return {"status": "ok", "version": app.version}


@app.post("/api/search")
def api_search(req: SearchRequest):
    """
    Text-to-image semantic search.
    Returns a ranked list of matching assets.
    """
    try:
        from backend.retrieval.search import semantic_search
        return semantic_search(
            query=req.query,
            category_filter=req.category,
            adapter=_get_adapter(),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/recommend")
def api_recommend(req: RecommendRequest):
    """
    Given an asset ID, return similar images from the database.
    """
    try:
        from backend.retrieval.search import recommend
        return recommend(
            asset_id=req.id,
            limit=req.limit,
            adapter=_get_adapter(),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/upload-and-index")
async def api_upload_and_index(file: UploadFile = File(...)):
    """
    Accept an image upload, auto-caption it with BLIP, compute a CLIP embedding,
    and store it in the database.
    """
    try:
        # Save uploaded file
        filename = f"{uuid.uuid4().hex[:8]}_{file.filename}"
        file_path = os.path.join(UPLOAD_DIR, filename)
        with open(file_path, "wb") as buf:
            shutil.copyfileobj(file.file, buf)

        # Generate caption and embedding
        from backend.models.blip_model import get_caption
        from backend.models.clip_model import get_image_embedding
        from backend.db.database import init_db, insert_asset

        init_db()
        caption = get_caption(file_path)
        emb = get_image_embedding(file_path)

        asset_id = f"up_{uuid.uuid4().hex[:8]}"
        title = file.filename.rsplit(".", 1)[0].replace("_", " ").replace("-", " ").title()
        url = f"/static/uploads/{filename}"

        insert_asset(
            asset_id=asset_id,
            url=url,
            title=title,
            category="Uploaded",
            description=caption,
            embedding_bytes=emb.astype("float32").tobytes(),
        )

        return {
            "status": "indexed",
            "id": asset_id,
            "url": url,
            "title": title,
            "caption": caption,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/agent/chat")
def api_agent_chat(req: ChatRequest):
    """
    Multi-turn Q&A via the ReAct LLM agent.
    Pass a session_id to maintain conversation history across requests.
    """
    try:
        session_id = req.session_id or "default"
        agent = _get_agent(session_id)
        result = agent.run(req.message)
        return {
            "answer": result["answer"],
            "retrieved_assets": result["retrieved_assets"],
            "session_id": session_id,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.delete("/api/agent/session/{session_id}")
def api_reset_session(session_id: str):
    """Clear the conversation history for a given session."""
    if session_id in _agent_sessions:
        _agent_sessions[session_id].reset()
    return {"status": "reset", "session_id": session_id}
