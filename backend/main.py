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


@app.get("/api/assets/stats")
def api_get_asset_stats():
    """Return database counts (total, original, uploaded) quickly."""
    try:
        from backend.db.database import get_connection
        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM assets")
                total = cursor.fetchone()[0]
                cursor.execute("SELECT COUNT(*) FROM assets WHERE data_source = 'new_addition'")
                uploaded = cursor.fetchone()[0]
                original = total - uploaded
        return {
            "code": 200,
            "message": "success",
            "data": {
                "totalCount": total,
                "originalCount": original,
                "uploadedCount": uploaded
            }
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/assets/recent")
def api_get_recent_assets(offset: int = 0, limit: int = 6):
    """Return paginated recently added assets, with S3 pre-signed URLs generated."""
    try:
        from backend.db.database import get_connection
        from psycopg2.extras import RealDictCursor
        from backend.retrieval.search import get_presigned_url

        with get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # Get all uploaded assets (new_addition)
                cursor.execute(
                    "SELECT id, url, title, category, description, base_code, subject_type, shooting_year, copyright, data_source, embedding FROM assets WHERE data_source = 'new_addition'"
                )
                uploaded = [dict(r) for r in cursor.fetchall()]
                # Reverse to show latest first
                uploaded.reverse()

                num_uploaded = len(uploaded)

                if offset >= num_uploaded:
                    orig_offset = offset - num_uploaded
                    cursor.execute(
                        "SELECT id, url, title, category, description, base_code, subject_type, shooting_year, copyright, data_source, embedding "
                        "FROM assets WHERE data_source != 'new_addition' OR data_source IS NULL "
                        "ORDER BY id DESC LIMIT %s OFFSET %s",
                        (limit, orig_offset)
                    )
                    recent = [dict(r) for r in cursor.fetchall()]
                else:
                    recent_uploaded = uploaded[offset : offset + limit]
                    needed_original = limit - len(recent_uploaded)

                    if needed_original > 0:
                        cursor.execute(
                            "SELECT id, url, title, category, description, base_code, subject_type, shooting_year, copyright, data_source, embedding "
                            "FROM assets WHERE data_source != 'new_addition' OR data_source IS NULL "
                            "ORDER BY id DESC LIMIT %s OFFSET 0",
                            (needed_original,)
                        )
                        recent_original = [dict(r) for r in cursor.fetchall()]
                    else:
                        recent_original = []

                    recent = recent_uploaded + recent_original

        # Perform visual stacking of highly similar assets
        import numpy as np
        SIMILARITY_THRESHOLD = 0.85
        recent_stacked = []
        primary_embeddings = []
        primary_indices = []

        for item in recent:
            emb_bytes = item.pop("embedding", None)
            if emb_bytes is None:
                item["stacked_assets"] = []
                recent_stacked.append(item)
                primary_embeddings.append(None)
                primary_indices.append(len(recent_stacked) - 1)
                continue

            try:
                emb = np.frombuffer(emb_bytes, dtype=np.float32)
            except Exception:
                item["stacked_assets"] = []
                recent_stacked.append(item)
                primary_embeddings.append(None)
                primary_indices.append(len(recent_stacked) - 1)
                continue

            is_stacked = False
            for p_idx, p_emb in zip(primary_indices, primary_embeddings):
                if p_emb is None:
                    continue
                sim = float(np.dot(emb, p_emb))
                if sim >= SIMILARITY_THRESHOLD:
                    if "stacked_assets" not in recent_stacked[p_idx]:
                        recent_stacked[p_idx]["stacked_assets"] = []
                    recent_stacked[p_idx]["stacked_assets"].append(item)
                    is_stacked = True
                    break

            if not is_stacked:
                item["stacked_assets"] = []
                recent_stacked.append(item)
                primary_embeddings.append(emb)
                primary_indices.append(len(recent_stacked) - 1)

        # Generate pre-signed S3 URLs for primary and stacked assets
        for asset in recent_stacked:
            url = asset["url"]
            if url.startswith("http") and (".s3." in url or "s3.amazonaws.com" in url):
                asset["url"] = get_presigned_url(url)
            
            if "stacked_assets" in asset:
                for sub_asset in asset["stacked_assets"]:
                    sub_url = sub_asset["url"]
                    if sub_url.startswith("http") and (".s3." in sub_url or "s3.amazonaws.com" in sub_url):
                        sub_asset["url"] = get_presigned_url(sub_url)

        # Calculate total available database rows for has_more computation
        total_original = 0
        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT COUNT(*) FROM assets WHERE data_source != 'new_addition' OR data_source IS NULL"
                )
                total_original = cursor.fetchone()[0]

        total_available = num_uploaded + total_original
        next_offset = offset + len(recent)
        has_more = next_offset < total_available

        return {
            "code": 200,
            "message": "success",
            "data": recent_stacked,
            "next_offset": next_offset,
            "has_more": has_more
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/assets/{asset_id}")
def api_get_asset(asset_id: str):
    """Return a single asset by its ID, with pre-signed S3 URL if applicable."""
    try:
        from backend.db.database import get_asset_by_id
        asset = get_asset_by_id(asset_id)
        if not asset:
            raise HTTPException(status_code=404, detail="Asset not found")

        from backend.retrieval.search import get_presigned_url
        url = asset["url"]
        if url.startswith("http") and (".s3." in url or "s3.amazonaws.com" in url):
            asset["url"] = get_presigned_url(url)

        return {"code": 200, "message": "success", "data": asset}
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

            emb_bytes = emb.astype("float32").tobytes()
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

        import urllib.parse
        all_assets = get_all_assets()
        existing_keys = set()
        for asset in all_assets:
            url = asset.get("url")
            if url:
                try:
                    decoded_url = urllib.parse.unquote(url)
                    parsed = urllib.parse.urlparse(decoded_url)
                    path = parsed.path.lstrip("/")
                    if path.startswith(f"{s3_bucket}/"):
                        path = path[len(s3_bucket) + 1:]
                    existing_keys.add(path)
                except Exception:
                    pass

        missing_keys = []
        region_str = f".{s3_region}" if s3_region else ""
        for key in all_keys:
            if key not in existing_keys:
                s3_url = f"https://{s3_bucket}.s3{region_str}.amazonaws.com/{key}"
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
    Multi-turn Q&A via the ReAct LLM agent (proxied to the algorithm service)
    with database vector search fallback and S3 pre-signed URL handling.
    """
    try:
        import requests
        retrieved_assets = []
        
        # 1. Try to get response from algorithm container ReAct agent
        try:
            resp = requests.post(
                f"{ALGO_API_BASE}/api/algo/agent/chat",
                json={"message": req.message, "session_id": req.session_id},
                timeout=25,
            )
            resp.raise_for_status()
            res_json = resp.json()
            if res_json.get("code") == 200:
                chat_data = res_json.get("data", {})
                answer = chat_data.get("answer", "")
                retrieved_assets = chat_data.get("retrieved_assets", [])
                session_id = chat_data.get("session_id", req.session_id)
            else:
                answer = f"[LLM unavailable: {res_json.get('message')}]"
                session_id = req.session_id
        except Exception as exc:
            answer = f"[LLM unavailable: {str(exc)}]"
            session_id = req.session_id

        # 2. If ReAct agent didn't return search results, perform fallback semantic search
        # We rely strictly on the similarity score threshold (>= 0.25) to distinguish conversational chats
        # (which yield low score noise, e.g., < 0.23) from actual semantic image search queries.
        if not retrieved_assets:
            try:
                from backend.retrieval.search import semantic_search
                search_results = semantic_search(query=req.message, adapter=True)
                # Only keep assets with a similarity score of >= 0.25 (meaningful match)
                retrieved_assets = [asset for asset in search_results if asset.get("score", 0) >= 0.25][:8]
            except Exception as search_exc:
                print(f"[CHAT SEARCH WARN] Fallback semantic search failed: {search_exc}")

        # 3. Generate pre-signed S3 URLs for all returned assets
        try:
            from backend.retrieval.search import get_presigned_url
            for asset in retrieved_assets:
                url = asset.get("url")
                if url and url.startswith("http") and (".s3." in url or "s3.amazonaws.com" in url):
                    asset["url"] = get_presigned_url(url)
        except Exception as url_exc:
            print(f"[CHAT URL WARN] Pre-signed URL generation failed: {url_exc}")

        return {
            "code": 200,
            "message": "success",
            "data": {
                "answer": answer,
                "retrieved_assets": retrieved_assets,
                "session_id": session_id
            }
        }
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


def _project_embeddings_pca(embeddings: np.ndarray) -> np.ndarray:
    if len(embeddings) < 2:
        return np.zeros((len(embeddings), 2))
    mean = np.mean(embeddings, axis=0)
    centered = embeddings - mean
    cov = np.cov(centered, rowvar=False)
    evals, evecs = np.linalg.eigh(cov)
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:, idx]
    top2_evecs = evecs[:, :2]
    projected = np.dot(centered, top2_evecs)
    p_min = np.min(projected, axis=0)
    p_max = np.max(projected, axis=0)
    denom = p_max - p_min
    denom[denom == 0] = 1.0
    projected = -100.0 + 200.0 * (projected - p_min) / denom
    return projected


@app.get("/api/evaluate")
def api_evaluate_dashboard():
    """
    Run evaluation on the golden test set and return baseline vs adapted metrics.
    Uses ultra-fast vectorized in-memory NumPy matrix evaluation to return within ~0.5s.
    """
    try:
        from backend.evaluation.evaluate import load_golden_test_set, DEFAULT_GOLDEN_PATH, _average_precision, _dcg
        from backend.retrieval.search import get_text_embedding_from_algo, apply_adapter_from_algo
        from backend.db.database import get_connection, get_all_assets_with_embeddings
        from psycopg2.extras import RealDictCursor
        
        golden_queries = []
        # 1. Try file
        if os.path.exists(DEFAULT_GOLDEN_PATH):
            try:
                with open(DEFAULT_GOLDEN_PATH, "r", encoding="utf-8") as f:
                    golden_queries = json.load(f)
            except Exception:
                pass
        
        # 2. Try golden_test_set table in database
        if not golden_queries:
            try:
                with get_connection() as conn:
                    with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                        cursor.execute("SELECT asset_id, caption FROM golden_test_set LIMIT 20")
                        db_rows = cursor.fetchall()
                        for row in db_rows:
                            golden_queries.append({
                                "asset_id": row["asset_id"],
                                "caption": row["caption"],
                                "query": row["caption"],
                                "relevant_ids": [row["asset_id"]]
                            })
            except Exception:
                pass

        # 3. Dynamic fallback: sample from assets table metadata (limit 15 for fast response)
        if not golden_queries:
            try:
                with get_connection() as conn:
                    with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                        cursor.execute("SELECT id, title, category, description, base_code FROM assets WHERE description IS NOT NULL AND description != '' LIMIT 15")
                        rows = cursor.fetchall()
                        if not rows:
                            cursor.execute("SELECT id, title, category, description, base_code FROM assets LIMIT 15")
                            rows = cursor.fetchall()
                            
                        for row in rows:
                            q_text = row.get("description") or row.get("title") or f"{row.get('category', 'Archive')} {row.get('base_code', '')}".strip()
                            if q_text:
                                golden_queries.append({
                                    "asset_id": row["id"],
                                    "caption": q_text,
                                    "query": q_text,
                                    "relevant_ids": [row["id"]]
                                })
            except Exception as exc:
                print(f"[EVAL WARN] Fallback asset query generation failed: {exc}")

        # Limit max queries for speed
        golden_queries = golden_queries[:20]

        # Check adapter status across all possible paths
        possible_adapter_paths = [
            os.path.join(BASE_DIR, "static", "models", "adapter.pth"),
            os.path.join(os.path.dirname(BASE_DIR), "backend", "static", "models", "adapter.pth"),
            "/app/backend/static/models/adapter.pth",
            "/app/static/models/adapter.pth"
        ]
        adapter_path = None
        for p in possible_adapter_paths:
            if os.path.exists(p):
                adapter_path = p
                break
                
        adapter_loaded = adapter_path is not None

        if not golden_queries:
            return {
                "code": 200,
                "message": "success",
                "data": {
                    "has_golden": False,
                    "queries_count": 0,
                    "adapter_loaded": adapter_loaded,
                    "baseline": {"map": 0, "ndcg": 0},
                    "adapted": {"map": 0, "ndcg": 0},
                    "queries": []
                }
            }

        # Load all asset embeddings ONCE into memory matrix
        all_assets = get_all_assets_with_embeddings()
        
        asset_ids = []
        raw_embs = []
        for a in all_assets:
            if a.get("embedding") is not None:
                try:
                    emb = np.frombuffer(a["embedding"], dtype=np.float32)
                    if len(emb) == 512:
                        raw_embs.append(emb)
                        asset_ids.append(a["id"])
                except Exception:
                    pass
                    
        if not raw_embs:
            return {
                "code": 200,
                "message": "success",
                "data": {
                    "has_golden": True,
                    "queries_count": len(golden_queries),
                    "adapter_loaded": adapter_loaded,
                    "baseline": {"map": 0, "ndcg": 0},
                    "adapted": {"map": 0, "ndcg": 0},
                    "queries": []
                }
            }
            
        baseline_matrix = np.vstack(raw_embs).astype("float32") # [N, 512]
        
        # Adapt database embeddings ONCE if adapter is loaded
        if adapter_loaded:
            try:
                adapted_matrix = apply_adapter_from_algo(baseline_matrix, adapter_path)
            except Exception as e:
                print(f"[EVAL WARN] Failed to apply adapter to matrix: {e}")
                adapted_matrix = baseline_matrix
        else:
            adapted_matrix = baseline_matrix

        queries_detail = []
        for item in golden_queries:
            q = item["query"]
            relevant = set(item["relevant_ids"])
            
            try:
                # Text embedding (single call)
                q_emb = get_text_embedding_from_algo(q)
                
                # Baseline similarity
                base_scores = np.dot(baseline_matrix, q_emb)
                base_sort_idx = np.argsort(base_scores)[::-1]
                base_ranked = [asset_ids[idx] for idx in base_sort_idx]
                base_ap = _average_precision(base_ranked, relevant)
                ideal_dcg = _dcg(list(relevant)[:10], relevant)
                base_ndcg = _dcg(base_ranked[:10], relevant) / ideal_dcg if ideal_dcg > 0 else 0.0
                
                # Adapted similarity (Text query stays in CLIP text space, database images in Adapted space)
                if adapter_loaded:
                    adapt_scores = np.dot(adapted_matrix, q_emb)
                    adapt_sort_idx = np.argsort(adapt_scores)[::-1]
                    adapt_ranked = [asset_ids[idx] for idx in adapt_sort_idx]
                    adapt_ap = _average_precision(adapt_ranked, relevant)
                    adapt_ndcg = _dcg(adapt_ranked[:10], relevant) / ideal_dcg if ideal_dcg > 0 else 0.0
                else:
                    adapt_ranked = base_ranked
                    adapt_ap = base_ap
                    adapt_ndcg = base_ndcg
                    
                base_rank = -1
                for rank, r_id in enumerate(base_ranked, start=1):
                    if r_id in relevant:
                        base_rank = rank
                        break
                
                adapt_rank = -1
                for rank, r_id in enumerate(adapt_ranked, start=1):
                    if r_id in relevant:
                        adapt_rank = rank
                        break
                
                queries_detail.append({
                    "query": q,
                    "relevant_count": len(relevant),
                    "baseline": {
                        "ap": round(base_ap, 4),
                        "ndcg": round(base_ndcg, 4),
                        "first_rank": base_rank
                    },
                    "adapted": {
                        "ap": round(adapt_ap, 4),
                        "ndcg": round(adapt_ndcg, 4),
                        "first_rank": adapt_rank
                    }
                })
            except Exception as q_err:
                print(f"[EVAL WARN] Evaluation for query '{q}' failed: {q_err}")

        base_map = float(np.mean([q["baseline"]["ap"] for q in queries_detail])) if queries_detail else 0.0
        base_ndcg = float(np.mean([q["baseline"]["ndcg"] for q in queries_detail])) if queries_detail else 0.0
        
        adapt_map = float(np.mean([q["adapted"]["ap"] for q in queries_detail])) if queries_detail else 0.0
        adapt_ndcg = float(np.mean([q["adapted"]["ndcg"] for q in queries_detail])) if queries_detail else 0.0
        
        return {
            "code": 200,
            "message": "success",
            "data": {
                "has_golden": True,
                "queries_count": len(queries_detail),
                "adapter_loaded": adapter_loaded,
                "baseline": {
                    "map": round(base_map, 4),
                    "ndcg": round(base_ndcg, 4)
                },
                "adapted": {
                    "map": round(adapt_map, 4),
                    "ndcg": round(adapt_ndcg, 4)
                },
                "queries": queries_detail
            }
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/evaluate/projection")
def api_evaluate_projection():
    """
    Project all database asset embeddings into 2D spaces for baseline and adapted models
    using zero-dependency NumPy PCA.
    """
    try:
        from backend.db.database import get_all_assets_with_embeddings
        from backend.retrieval.search import apply_adapter_from_algo, get_presigned_url
        
        rows = get_all_assets_with_embeddings()
        
        valid_assets = []
        raw_embs = []
        for r in rows:
            if r.get("embedding") is not None:
                try:
                    emb = np.frombuffer(r["embedding"], dtype=np.float32)
                    if len(emb) == 512:
                        raw_embs.append(emb)
                        valid_assets.append(r)
                except Exception:
                    pass
                    
        if not raw_embs:
            return {
                "code": 200,
                "message": "success",
                "data": {
                    "adapter_loaded": False,
                    "points": []
                }
            }
            
        baseline_matrix = np.vstack(raw_embs).astype("float32")
        baseline_proj = _project_embeddings_pca(baseline_matrix)
        
        possible_adapter_paths = [
            os.path.join(BASE_DIR, "static", "models", "adapter.pth"),
            os.path.join(os.path.dirname(BASE_DIR), "backend", "static", "models", "adapter.pth"),
            "/app/backend/static/models/adapter.pth",
            "/app/static/models/adapter.pth"
        ]
        adapter_path = None
        for p in possible_adapter_paths:
            if os.path.exists(p):
                adapter_path = p
                break
                
        adapter_loaded = adapter_path is not None
        
        if adapter_loaded:
            try:
                adapted_matrix = apply_adapter_from_algo(baseline_matrix, adapter_path)
                adapted_proj = _project_embeddings_pca(adapted_matrix)
            except Exception as e:
                print(f"[PROJECTION WARN] Failed to apply adapter: {e}")
                adapted_proj = baseline_proj
        else:
            adapted_proj = baseline_proj
            
        points = []
        for i, row in enumerate(valid_assets):
            url = row["url"]
            if url and url.startswith("http") and (".s3." in url or "s3.amazonaws.com" in url):
                url = get_presigned_url(url)
                
            points.append({
                "id": row["id"],
                "title": row["title"] or "Untitled",
                "category": row["category"] or "Uncategorized",
                "url": url,
                "baseline": {
                    "x": float(baseline_proj[i, 0]),
                    "y": float(baseline_proj[i, 1])
                },
                "adapted": {
                    "x": float(adapted_proj[i, 0]),
                    "y": float(adapted_proj[i, 1])
                }
            })
            
        return {
            "code": 200,
            "message": "success",
            "data": {
                "adapter_loaded": adapter_loaded,
                "points": points
            }
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))



# ---------------------------------------------------------------------------
# Serve React SPA static files (Single Origin Deployment)
# ---------------------------------------------------------------------------
react_dist_dir = os.path.join(os.path.dirname(BASE_DIR), "frontend-react", "dist")
if os.path.exists(react_dist_dir):
    app.mount("/", StaticFiles(directory=react_dist_dir, html=True), name="react")
