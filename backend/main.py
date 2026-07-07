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

import json
import os
import shutil
import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import BaseModel
from typing import Optional, Any

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class UpdateCaptionRequest(BaseModel):
    caption: str


class GoldenEntryRequest(BaseModel):
    asset_id: str
    caption: str

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
# Global Exception Handlers for {code, message, data} standard format
# ---------------------------------------------------------------------------

@app.exception_handler(StarletteHTTPException)
async def custom_http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "code": exc.status_code,
            "message": exc.detail,
            "data": None
        }
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content={
            "code": 422,
            "message": f"Validation Error: {exc.errors()}",
            "data": None
        }
    )

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    import traceback
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={
            "code": 500,
            "message": str(exc),
            "data": None
        }
    )

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
    return {"code": 200, "message": "success", "data": {"status": "ok", "version": app.version}}


@app.get("/api/assets")
def api_get_all_assets():
    """Return all assets in the database."""
    try:
        from backend.db.database import get_all_assets
        return {"code": 200, "message": "success", "data": get_all_assets()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.put("/api/assets/{asset_id}/caption")
def api_update_caption(asset_id: str, req: UpdateCaptionRequest):
    """Update caption/description for a specific asset."""
    try:
        from backend.db.database import get_asset_by_id, update_asset_description
        asset = get_asset_by_id(asset_id)
        if not asset:
            raise HTTPException(status_code=404, detail="Asset not found")
        update_asset_description(asset_id, req.caption)
        return {"code": 200, "message": "success", "data": {"status": "updated", "id": asset_id}}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/golden")
def api_get_golden():
    """Load the golden test set from file."""
    try:
        golden_path = os.path.join(os.path.dirname(BASE_DIR), "golden_test_set.json")
        if os.path.exists(golden_path):
            with open(golden_path, "r", encoding="utf-8") as f:
                return {"code": 200, "message": "success", "data": json.load(f)}
        return {"code": 200, "message": "success", "data": []}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/golden")
def api_add_golden(req: GoldenEntryRequest):
    """Add a verified image-caption pair to the golden test set."""
    try:
        golden_path = os.path.join(os.path.dirname(BASE_DIR), "golden_test_set.json")
        golden = []
        if os.path.exists(golden_path):
            with open(golden_path, "r", encoding="utf-8") as f:
                try:
                    golden = json.load(f)
                except Exception:
                    golden = []

        # Check if this asset already has a golden entry
        existing_ids = {e["asset_id"] for e in golden}
        if req.asset_id not in existing_ids:
            golden.append({
                "asset_id": req.asset_id,
                "caption": req.caption,
                "query": req.caption,
                "relevant_ids": [req.asset_id],
            })
            os.makedirs(os.path.dirname(golden_path), exist_ok=True)
            with open(golden_path, "w", encoding="utf-8") as f:
                json.dump(golden, f, indent=2, ensure_ascii=False)
            
            # Also insert into database golden table for sync
            try:
                from backend.db.database import insert_golden_entry
                import uuid
                insert_golden_entry(str(uuid.uuid4()), req.asset_id, req.caption)
            except Exception:
                pass
            
            return {"code": 200, "message": "success", "data": {"status": "added", "count": len(golden)}}
        else:
            return {"code": 200, "message": "success", "data": {"status": "exists", "count": len(golden)}}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/search")
def api_search(req: SearchRequest):
    """
    Text-to-image semantic search.
    Returns a ranked list of matching assets.
    """
    try:
        from backend.retrieval.search import semantic_search
        results = semantic_search(
            query=req.query,
            category_filter=req.category,
            adapter=_get_adapter(),
        )
        return {"code": 200, "message": "success", "data": results}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/recommend")
def api_recommend(req: RecommendRequest):
    """
    Given an asset ID, return similar images from the database.
    """
    try:
        from backend.retrieval.search import recommend
        results = recommend(
            asset_id=req.id,
            limit=req.limit,
            adapter=_get_adapter(),
        )
        return {"code": 200, "message": "success", "data": results}
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
            "code": 200,
            "message": "success",
            "data": {
                "status": "indexed",
                "id": asset_id,
                "url": url,
                "title": title,
                "caption": caption,
            }
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


@app.delete("/api/agent/session/{session_id}")
def api_reset_session(session_id: str):
    """Clear the conversation history for a given session."""
    if session_id in _agent_sessions:
        _agent_sessions[session_id].reset()
    return {"code": 200, "message": "success", "data": {"status": "reset", "session_id": session_id}}
