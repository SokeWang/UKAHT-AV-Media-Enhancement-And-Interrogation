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
import threading
import numpy as np
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile, BackgroundTasks
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

# Initialize database tables and user at application startup
from backend.db.database import init_db
try:
    init_db()
except Exception as _e:
    print(f"Error initializing database at startup: {_e}")

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

class LoginRequest(BaseModel):
    username: str
    password: str


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
# ---------------------------------------------------------------------------
# Configured Service URL
# ---------------------------------------------------------------------------
ALGO_API_BASE = os.getenv("ALGO_API_BASE", "http://localhost:8001")


class SqlFilterRequest(BaseModel):
    category: Optional[str] = None
    keyword: Optional[str] = None
    base_code: Optional[str] = None
    subject_type: Optional[str] = None
    shooting_year: Optional[str] = None
    copyright: Optional[str] = None
    data_source: Optional[str] = None


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health_check():
    """Quick liveness check."""
    return {"code": 200, "message": "success", "data": {"status": "ok", "version": app.version}}


@app.post("/api/auth/login")
def api_login(req: LoginRequest):
    """Authenticate user credentials."""
    try:
        from backend.db.database import verify_user
        if verify_user(req.username, req.password):
            return {"code": 200, "message": "success", "data": {"authenticated": True}}
        else:
            raise HTTPException(status_code=401, detail="Invalid username or password")
    except HTTPException as he:
        raise he
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/assets")
def api_get_all_assets():
    """Return all assets in the database."""
    try:
        from backend.db.database import get_all_assets
        assets = get_all_assets()
        from backend.retrieval.search import get_presigned_url
        for asset in assets:
            url = asset["url"]
            if url.startswith("http") and (".s3." in url or "s3.amazonaws.com" in url):
                asset["url"] = get_presigned_url(url)
        return {"code": 200, "message": "success", "data": assets}
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
        # Passes adapter=True so that the query and database embeddings are adapted via the algorithm service if trained
        results = semantic_search(
            query=req.query,
            category_filter=req.category,
            adapter=True,
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
            adapter=True,
        )
        return {"code": 200, "message": "success", "data": results}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/sql-filter")
def api_sql_filter(req: SqlFilterRequest):
    """
    Structured metadata SQL filter endpoint. Used primarily by the LangChain agent.
    """
    try:
        from backend.retrieval.search import sql_metadata_filter
        results = sql_metadata_filter(
            category=req.category,
            keyword=req.keyword,
            base_code=req.base_code,
            subject_type=req.subject_type,
            shooting_year=req.shooting_year,
            copyright=req.copyright,
            data_source=req.data_source,
        )
        return {"code": 200, "message": "success", "data": results}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


def extract_nested_zips_recursively(zip_path: str, extract_to: str) -> list[str]:
    import zipfile
    to_process = [zip_path]
    os.makedirs(extract_to, exist_ok=True)
    
    while to_process:
        curr_zip = to_process.pop(0)
        try:
            with zipfile.ZipFile(curr_zip, 'r') as zf:
                zf.extractall(path=extract_to)
        except Exception:
            continue
            
        # Scan for nested zips
        for root, dirs, files in os.walk(extract_to):
            for file in files:
                file_path = os.path.join(root, file)
                if file.lower().endswith(".zip") and file_path not in to_process:
                    to_process.append(file_path)
                    
    # Clean up zip files
    for root, dirs, files in os.walk(extract_to):
        for file in files:
            if file.lower().endswith(".zip"):
                try:
                    os.remove(os.path.join(root, file))
                except Exception:
                    pass
                    
    # Collect all supported image paths
    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
    image_paths = []
    for root, dirs, files in os.walk(extract_to):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                image_paths.append(os.path.join(root, file))
                
    return image_paths


@app.post("/api/upload-and-index")
async def api_upload_and_index(file: UploadFile = File(...)):
    """
    Accept an image or ZIP file upload, recursively extract zips, compute captions/embeddings
    via the algorithm service, upload to S3 if configured, and store it in the database.
    """
    try:
        import requests
        import numpy as np
        from pathlib import Path
        from backend.db.database import init_db, insert_asset
        from backend.ingest.ingest_real_data import extract_metadata_via_llm
        init_db()

        # Save uploaded file temporarily
        filename = f"{uuid.uuid4().hex[:8]}_{file.filename}"
        file_path = os.path.join(UPLOAD_DIR, filename)
        with open(file_path, "wb") as buf:
            shutil.copyfileobj(file.file, buf)

        # Check if the uploaded file is a ZIP archive
        is_zip = file.filename.lower().endswith(".zip")
        s3_bucket = os.getenv("UKAHT_S3_BUCKET")
        s3_region = os.getenv("UKAHT_S3_REGION")
        upload_id = uuid.uuid4().hex[:8]

        indexed_assets = []

        if is_zip:
            extract_to = os.path.join(UPLOAD_DIR, f"zip_temp_{upload_id}")
            image_paths = extract_nested_zips_recursively(file_path, extract_to)
            
            if not s3_bucket:
                perm_base_dir = os.path.join(UPLOAD_DIR, "unzipped", upload_id)
                os.makedirs(perm_base_dir, exist_ok=True)
                
            for img_path in image_paths:
                rel_path = os.path.relpath(img_path, extract_to)
                
                # Generate caption and embedding via algorithm service
                caption_resp = requests.post(
                    f"{ALGO_API_BASE}/api/algo/caption",
                    params={"path": img_path},
                    timeout=30
                )
                caption_resp.raise_for_status()
                caption = caption_resp.json()["data"]["caption"]

                embed_resp = requests.post(
                    f"{ALGO_API_BASE}/api/algo/embed/image",
                    params={"path": img_path},
                    timeout=30
                )
                embed_resp.raise_for_status()
                emb_list = embed_resp.json()["data"]["embedding"]
                emb = np.array(emb_list, dtype=np.float32)

                # Check for duplicate image in the database
                from backend.db.database import check_duplicate_image, get_asset_by_id
                emb_bytes = emb.astype("float32").tobytes()
                duplicate_id = check_duplicate_image(emb_bytes)
                if duplicate_id:
                    existing = get_asset_by_id(duplicate_id)
                    if existing:
                        indexed_assets.append({
                            "id": existing["id"],
                            "url": existing["url"],
                            "title": existing["title"],
                            "caption": existing["description"]
                        })
                    continue

                # Determine URL and handle S3 upload if configured
                if s3_bucket:
                    import boto3
                    s3_client = boto3.client("s3", region_name=s3_region) if s3_region else boto3.client("s3")
                    s3_key = f"uploads/{upload_id}/{rel_path}"
                    s3_client.upload_file(img_path, s3_bucket, s3_key)
                    
                    region_str = f".{s3_region}" if s3_region else ""
                    url = f"https://{s3_bucket}.s3{region_str}.amazonaws.com/{s3_key}"
                else:
                    dest_path = os.path.join(perm_base_dir, rel_path)
                    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                    shutil.copy2(img_path, dest_path)
                    url = f"/static/uploads/unzipped/{upload_id}/{rel_path}"

                asset_id = f"up_{uuid.uuid4().hex[:8]}"
                title = Path(img_path).stem.replace("_", " ").replace("-", " ").title()
                
                # Extract metadata using the relative path inside the zip archive to preserve folder structure
                meta = extract_metadata_via_llm(rel_path)
                category = meta["subject_type"] or Path(img_path).parent.name.replace("_", " ").title()

                insert_asset(
                    asset_id=asset_id,
                    url=url,
                    title=title,
                    category=category,
                    description=caption,
                    embedding_bytes=emb_bytes,
                    base_code=meta["base_code"],
                    subject_type=meta["subject_type"],
                    shooting_year=meta["shooting_year"],
                    copyright=meta["copyright"],
                    data_source=meta["data_source"]
                )
                
                indexed_assets.append({
                    "id": asset_id,
                    "url": url,
                    "title": title,
                    "caption": caption
                })

            # Clean up temporary zip and extraction folders
            try:
                os.remove(file_path)
                shutil.rmtree(extract_to)
            except Exception:
                pass
                
            return {
                "code": 200,
                "message": "success",
                "data": {
                    "status": "indexed_batch",
                    "count": len(indexed_assets),
                    "assets": indexed_assets
                }
            }
        else:
            # Single image upload
            # Call algorithm service for BLIP captioning
            caption_resp = requests.post(
                f"{ALGO_API_BASE}/api/algo/caption",
                params={"path": file_path},
                timeout=30
            )
            caption_resp.raise_for_status()
            caption = caption_resp.json()["data"]["caption"]

            # Call algorithm service for CLIP image embedding
            embed_resp = requests.post(
                f"{ALGO_API_BASE}/api/algo/embed/image",
                params={"path": file_path},
                timeout=30
            )
            embed_resp.raise_for_status()
            emb_list = embed_resp.json()["data"]["embedding"]
            emb = np.array(emb_list, dtype=np.float32)

            # Check for duplicate image in the database
            from backend.db.database import check_duplicate_image, get_asset_by_id
            emb_bytes = emb.astype("float32").tobytes()
            duplicate_id = check_duplicate_image(emb_bytes)
            if duplicate_id:
                try:
                    os.remove(file_path)
                except Exception:
                    pass
                existing = get_asset_by_id(duplicate_id)
                return {
                    "code": 200,
                    "message": "success",
                    "data": {
                        "status": "already_exists",
                        "id": existing["id"],
                        "url": existing["url"],
                        "title": existing["title"],
                        "caption": existing["description"]
                    }
                }

            asset_id = f"up_{uuid.uuid4().hex[:8]}"
            title = file.filename.rsplit(".", 1)[0].replace("_", " ").replace("-", " ").title()

            if s3_bucket:
                import boto3
                s3_client = boto3.client("s3", region_name=s3_region) if s3_region else boto3.client("s3")
                s3_key = f"uploads/{upload_id}/{filename}"
                s3_client.upload_file(file_path, s3_bucket, s3_key)
                
                region_str = f".{s3_region}" if s3_region else ""
                url = f"https://{s3_bucket}.s3{region_str}.amazonaws.com/{s3_key}"
                
                try:
                    os.remove(file_path)
                except Exception:
                    pass
            else:
                url = f"/static/uploads/{filename}"

            # Pass the filename directly as relative path to preserve single metadata extraction fallback
            meta = extract_metadata_via_llm(file.filename)
            category = meta["subject_type"] or "Uploaded"

            insert_asset(
                asset_id=asset_id,
                url=url,
                title=title,
                category=category,
                description=caption,
                embedding_bytes=emb_bytes,
                base_code=meta["base_code"],
                subject_type=meta["subject_type"],
                shooting_year=meta["shooting_year"],
                copyright=meta["copyright"],
                data_source=meta["data_source"]
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



_sync_lock = threading.Lock()
_sync_status = {
    "status": "idle",
    "processed": 0,
    "total": 0,
    "message": "System idle"
}

def perform_s3_sync():
    global _sync_status
    import boto3
    import requests
    from backend.db.database import get_all_assets, insert_asset, check_duplicate_image
    from backend.ingest.ingest_real_data import scan_images_s3, extract_metadata_via_llm

    s3_bucket = os.getenv("UKAHT_S3_BUCKET")
    s3_region = os.getenv("UKAHT_S3_REGION")

    if not s3_bucket:
        with _sync_lock:
            _sync_status = {
                "status": "error",
                "processed": 0,
                "total": 0,
                "message": "S3 bucket is not configured. Set UKAHT_S3_BUCKET."
            }
        return

    try:
        with _sync_lock:
            _sync_status["status"] = "scanning"
            _sync_status["message"] = f"Scanning S3 bucket '{s3_bucket}'..."

        all_keys = scan_images_s3(s3_bucket, region=s3_region)

        all_assets = get_all_assets()
        existing_urls = {asset["url"] for asset in all_assets if asset.get("url")}

        missing_keys = []
        region_str = f".{s3_region}" if s3_region else ""
        for key in all_keys:
            s3_url = f"https://{s3_bucket}.s3{region_str}.amazonaws.com/{key}"
            if s3_url not in existing_urls:
                missing_keys.append((key, s3_url))

        total_missing = len(missing_keys)
        with _sync_lock:
            _sync_status["status"] = "syncing"
            _sync_status["total"] = total_missing
            _sync_status["processed"] = 0
            _sync_status["message"] = f"Found {total_missing} missing assets in database. Starting ingestion..."

        if total_missing == 0:
            with _sync_lock:
                _sync_status["status"] = "success"
                _sync_status["message"] = "Database is already up to date with S3."
            return

        temp_dir = os.path.join(BASE_DIR, "static", "uploads", "s3_temp")
        os.makedirs(temp_dir, exist_ok=True)
        s3_client = boto3.client("s3", region_name=s3_region) if s3_region else boto3.client("s3")

        batch_size = 4
        for i in range(0, total_missing, batch_size):
            batch = missing_keys[i : i + batch_size]
            local_paths = []
            valid_batch_keys = []
            valid_s3_urls = []

            for key, s3_url in batch:
                ext = os.path.splitext(key)[1].lower()
                filename = f"{uuid.uuid4().hex[:12]}{ext}"
                local_path = os.path.join(temp_dir, filename)
                try:
                    s3_client.download_file(s3_bucket, key, local_path)
                    local_paths.append(local_path)
                    valid_batch_keys.append(key)
                    valid_s3_urls.append(s3_url)
                except Exception as exc:
                    print(f"[SYNC ERROR] Failed to download {key}: {exc}")

            if not local_paths:
                continue

            try:
                resp = requests.post(f"{ALGO_API_BASE}/api/algo/caption/batch", json={"paths": local_paths}, timeout=60)
                resp.raise_for_status()
                captions = resp.json()["data"]["captions"]
            except Exception as exc:
                print(f"[SYNC WARN] BLIP captioning failed: {exc}")
                captions = [""] * len(local_paths)

            try:
                resp = requests.post(f"{ALGO_API_BASE}/api/algo/embed/image/batch", json={"paths": local_paths}, timeout=60)
                resp.raise_for_status()
                embeddings_list = resp.json()["data"]["embeddings"]
                embeddings = [np.array(emb, dtype=np.float32) for emb in embeddings_list]
            except Exception as exc:
                print(f"[SYNC WARN] CLIP embedding failed: {exc}")
                embeddings = [np.zeros(512, dtype="float32")] * len(local_paths)

            for idx, (key, caption, emb, url) in enumerate(zip(valid_batch_keys, captions, embeddings, valid_s3_urls)):
                emb_bytes = emb.astype("float32").tobytes()
                duplicate_id = check_duplicate_image(emb_bytes)
                if duplicate_id:
                    try:
                        os.remove(local_paths[idx])
                    except Exception:
                        pass
                    with _sync_lock:
                        _sync_status["processed"] += 1
                    continue

                asset_id = f"ukaht_{uuid.uuid4().hex[:10]}"
                title_stem = Path(key).stem
                title = title_stem.replace("_", " ").replace("-", " ").title()

                meta = extract_metadata_via_llm(key)
                category = meta["subject_type"] or "S3 Ingested"

                insert_asset(
                    asset_id=asset_id,
                    url=url,
                    title=title,
                    category=category,
                    description=caption,
                    embedding_bytes=emb_bytes,
                    base_code=meta["base_code"],
                    subject_type=meta["subject_type"],
                    shooting_year=meta["shooting_year"],
                    copyright=meta["copyright"],
                    data_source="new_addition"
                )
                try:
                    os.remove(local_paths[idx])
                except Exception:
                    pass
                with _sync_lock:
                    _sync_status["processed"] += 1
                    _sync_status["message"] = f"Processed {_sync_status['processed']}/{total_missing} missing assets."

        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass

        with _sync_lock:
            _sync_status["status"] = "success"
            _sync_status["message"] = f"Successfully synchronized {total_missing} assets with S3."

    except Exception as e:
        with _sync_lock:
            _sync_status["status"] = "error"
            _sync_status["message"] = f"Sync failed: {str(e)}"

@app.post("/api/assets/sync")
def api_sync_assets(background_tasks: BackgroundTasks):
    global _sync_status
    with _sync_lock:
        if _sync_status["status"] in ["scanning", "syncing"]:
            return {
                "code": 200,
                "message": "Synchronization is already in progress.",
                "data": _sync_status
            }
        _sync_status = {
            "status": "scanning",
            "processed": 0,
            "total": 0,
            "message": "Starting synchronization process..."
        }
    background_tasks.add_task(perform_s3_sync)
    return {
        "code": 200,
        "message": "Sync started in background.",
        "data": _sync_status
    }

@app.get("/api/assets/sync/status")
def api_sync_status():
    global _sync_status
    with _sync_lock:
        return {
            "code": 200,
            "data": _sync_status
        }


@app.post("/api/agent/chat")
def api_agent_chat(req: ChatRequest):
    """
    Multi-turn Q&A via the ReAct LLM agent (proxied to the algorithm service).
    """
    try:
        import requests
        resp = requests.post(
            f"{ALGO_API_BASE}/api/algo/agent/chat",
            json={"message": req.message, "session_id": req.session_id},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.delete("/api/agent/session/{session_id}")
def api_reset_session(session_id: str):
    """Clear the conversation history for a given session (proxied to the algorithm service)."""
    try:
        import requests
        resp = requests.delete(
            f"{ALGO_API_BASE}/api/algo/agent/session/{session_id}",
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Serve React SPA static files (Single Origin Deployment)
# ---------------------------------------------------------------------------
react_dist_dir = os.path.join(os.path.dirname(BASE_DIR), "frontend-react", "dist")
if os.path.exists(react_dist_dir):
    app.mount("/", StaticFiles(directory=react_dist_dir, html=True), name="react")
