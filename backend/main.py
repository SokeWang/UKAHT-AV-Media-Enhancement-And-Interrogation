import os
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import shutil

from backend.functions import (
    semantic_search,
    index_uploaded_image,
    recommend_similar_images
)

# Setup Static Directories and Frontend Mount
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
FRONTEND_DIST_DIR = os.path.join(os.path.dirname(BASE_DIR), "frontend", "dist")

app = FastAPI(title="UKAHT Antarctic Media Interrogation System", version="1.0.0")

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Request Models
class SearchRequest(BaseModel):
    query: str

class RecommendRequest(BaseModel):
    id: str
    limit: Optional[int] = 4

# API Endpoints
@app.post("/api/search")
def api_search(req: SearchRequest):
    try:
        return semantic_search(req.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload-and-index")
async def api_upload_and_index(file: UploadFile = File(...)):
    try:
        upload_dir = os.path.join(STATIC_DIR, "uploads")
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        result = index_uploaded_image(file_path, file.filename)
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result.get("description", "Failed to index image"))
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/recommend")
def api_recommend(req: RecommendRequest):
    try:
        return recommend_similar_images(req.id, req.limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Ensure static media directories exist
os.makedirs(os.path.join(STATIC_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(STATIC_DIR, "models"), exist_ok=True)
os.makedirs(os.path.join(STATIC_DIR, "overlays"), exist_ok=True)

# Mount backend static media
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

DATASET_DIR = "/Users/wangpeidong/UKAHT-Project/UK Antarctic Heritage Trust Data"
if os.path.exists(DATASET_DIR):
    app.mount("/data", StaticFiles(directory=DATASET_DIR), name="data")

# Mount frontend static build files if the build directory exists
if os.path.exists(FRONTEND_DIST_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIST_DIR, html=True), name="frontend")
else:
    @app.get("/")
    def read_root():
        return {
            "message": "FastAPI is running. Frontend build directory not found. Please compile frontend using 'npm run build'."
        }
