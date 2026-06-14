import os
import glob
import sqlite3
import uuid
import numpy as np
from PIL import Image
import cv2

# Lazy loading of ML pipelines and models
_clip_classifier = None
_blip_model = None
_blip_processor = None
_clip_model = None
_clip_processor = None

def get_clip_classifier():
    global _clip_classifier
    if _clip_classifier is None:
        from transformers import pipeline
        _clip_classifier = pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch32")
    return _clip_classifier

def get_blip_captioner():
    global _blip_model, _blip_processor
    if _blip_model is None:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        _blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        _blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return _blip_processor, _blip_model

def get_clip_model():
    global _clip_model, _clip_processor
    if _clip_model is None:
        from transformers import CLIPModel, CLIPProcessor
        _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return _clip_model, _clip_processor

def get_image_embedding(image_source) -> np.ndarray:
    import torch
    model, processor = get_clip_model()
    if isinstance(image_source, str):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        if image_source.startswith("/static/"):
            image_source = os.path.join(base_dir, "static", image_source[len("/static/"):])
        img = Image.open(image_source).convert("RGB")
    elif isinstance(image_source, bytes):
        import io
        img = Image.open(io.BytesIO(image_source)).convert("RGB")
    else:
        img = image_source
    
    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        vision_outputs = model.vision_model(pixel_values=inputs['pixel_values'])
        pooled = vision_outputs[1]
        features = model.visual_projection(pooled)
    features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy()[0]

def get_text_embedding(text: str) -> np.ndarray:
    import torch
    model, processor = get_clip_model()
    inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        text_outputs = model.text_model(input_ids=inputs['input_ids'], attention_mask=inputs.get('attention_mask'))
        pooled_text = text_outputs[1]
        features = model.text_projection(pooled_text)
    features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy()[0]


def get_image_embeddings_batch(image_sources: list) -> list:
    import torch
    model, processor = get_clip_model()
    imgs = []
    for src in image_sources:
        if isinstance(src, str):
            base_dir = os.path.dirname(os.path.abspath(__file__))
            if src.startswith("/static/"):
                src = os.path.join(base_dir, "static", src[len("/static/"):])
            imgs.append(Image.open(src).convert("RGB"))
        elif isinstance(src, bytes):
            import io
            imgs.append(Image.open(io.BytesIO(src)).convert("RGB"))
        else:
            imgs.append(src)
            
    inputs = processor(images=imgs, return_tensors="pt", padding=True)
    with torch.no_grad():
        vision_outputs = model.vision_model(pixel_values=inputs['pixel_values'])
        pooled = vision_outputs[1]
        features = model.visual_projection(pooled)
    features = features / features.norm(dim=-1, keepdim=True)
    return [f for f in features.cpu().numpy()]


def get_blip_captions_batch(images: list) -> list:
    processor, model = get_blip_captioner()
    inputs = processor(images=images, return_tensors="pt", padding=True)
    out = model.generate(**inputs)
    return [processor.decode(o, skip_special_tokens=True) for o in out]


def classify_images_batch(images: list, candidate_labels: list) -> list:
    classifier = get_clip_classifier()
    pil_imgs = []
    for img in images:
        if isinstance(img, str):
            base_dir = os.path.dirname(os.path.abspath(__file__))
            if img.startswith("/static/"):
                img = os.path.join(base_dir, "static", img[len("/static/"):])
            pil_imgs.append(Image.open(img).convert("RGB"))
        else:
            pil_imgs.append(img)
            
    clip_res_list = classifier(pil_imgs, candidate_labels=candidate_labels)
    if isinstance(clip_res_list, dict):
        clip_res_list = [clip_res_list]
        
    batch_results = []
    for clip_res in clip_res_list:
        category = clip_res[0]["label"].capitalize()
        labels = [item["label"] for item in clip_res if item["score"] > 0.15]
        batch_results.append({
            "category": category,
            "labels": labels
        })
    return batch_results



def classify_image(image_path: str) -> dict:
    """
    Real CLIP + BLIP image classification and description baseline.
    """
    try:
        # Convert path if relative to static directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        if image_path.startswith("/static/"):
            image_path = os.path.join(base_dir, "static", image_path[len("/static/"):])

        img = Image.open(image_path).convert("RGB")
        classifier = get_clip_classifier()
        processor, model = get_blip_captioner()

        # 1. Zero-shot CLIP tagging
        candidate_labels = ["building", "landscape", "artifact", "equipment", "wildlife"]
        clip_res = classifier(img, candidate_labels=candidate_labels)
        category = clip_res[0]["label"].capitalize()
        labels = [item["label"] for item in clip_res if item["score"] > 0.15]

        # 2. BLIP captioning
        inputs = processor(img, return_tensors="pt")
        out = model.generate(**inputs)
        description = processor.decode(out[0], skip_special_tokens=True)

        return {
            "status": "success",
            "category": category,
            "labels": labels,
            "description": description
        }
    except Exception as e:
        return {
            "status": "error",
            "category": "Unknown",
            "labels": [],
            "description": f"Failed to run classification baseline: {str(e)}"
        }


# SQLite DB initialization for Vector/Semantic search baseline
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "db.sqlite")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS assets (
        id TEXT PRIMARY KEY,
        url TEXT,
        title TEXT,
        category TEXT,
        description TEXT,
        embedding BLOB
    )
    """)
    
    # Check if empty or mismatching dimension, populate/repopulate with seed data
    cursor.execute("SELECT embedding FROM assets LIMIT 1")
    row = cursor.fetchone()
    needs_repopulate = False
    if row:
        emb_len = len(np.frombuffer(row[0], dtype=np.float32))
        if emb_len != 512:
            print("Detected old embedding dimension. Dropping assets table for upgrade to CLIP...")
            cursor.execute("DROP TABLE assets")
            cursor.execute("""
            CREATE TABLE assets (
                id TEXT PRIMARY KEY,
                url TEXT,
                title TEXT,
                category TEXT,
                description TEXT,
                embedding BLOB
            )
            """)
            conn.commit()
            needs_repopulate = True
    else:
        needs_repopulate = True

    cursor.execute("SELECT COUNT(*) FROM assets")
    if cursor.fetchone()[0] == 0 or needs_repopulate:
        initial_assets = [
            {
                "id": "img_001",
                "url": "/static/images/mock_sample1.png",
                "title": "Bransfield House Front Facade",
                "category": "Building",
                "description": "Front facade of Bransfield House in Port Lockroy."
            },
            {
                "id": "img_002",
                "url": "/static/images/mock_sample2.png",
                "title": "Glacial Landscape and Mountains",
                "category": "Landscape",
                "description": "Wide panoramic view of mountains near Port Lockroy."
            },
            {
                "id": "img_003",
                "url": "/static/images/mock_sample3.png",
                "title": "Historical Artifact Can",
                "category": "Artefact",
                "description": "A historic food tin can from the kitchen shelves."
            }
        ]
        
        for item in initial_assets:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            rel_url = item["url"]
            if rel_url.startswith("/static/"):
                img_path = os.path.join(base_dir, "static", rel_url[len("/static/"):])
            else:
                img_path = os.path.join(base_dir, rel_url)
            try:
                emb = get_image_embedding(img_path).astype(np.float32).tobytes()
            except Exception as e:
                print(f"Failed to embed seed asset {img_path}: {e}")
                emb = np.zeros(512, dtype=np.float32).tobytes()
                
            cursor.execute(
                "INSERT OR REPLACE INTO assets (id, url, title, category, description, embedding) VALUES (?, ?, ?, ?, ?, ?)",
                (item["id"], item["url"], item["title"], item["category"], item["description"], emb)
            )
        conn.commit()
    conn.close()


def semantic_search(query: str) -> list:
    """
    Real CLIP text embedding + Cosine Similarity semantic search.
    """
    try:
        init_db()
        if not query.strip():
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT id, url, title, category, description FROM assets")
            rows = cursor.fetchall()
            conn.close()
            return [{
                "id": r[0], "url": r[1], "title": r[2], "category": r[3], "description": r[4], "score": 1.0
            } for r in rows]

        query_emb = get_text_embedding(query).astype(np.float32)

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT id, url, title, category, description, embedding FROM assets")
        rows = cursor.fetchall()
        conn.close()

        results = []
        for r in rows:
            emb = np.frombuffer(r[5], dtype=np.float32)
            dot = np.dot(query_emb, emb)
            norm_q = np.linalg.norm(query_emb)
            norm_e = np.linalg.norm(emb)
            score = float(dot / (norm_q * norm_e)) if (norm_q * norm_e) > 0 else 0.0

            results.append({
                "id": r[0],
                "url": r[1],
                "title": r[2],
                "category": r[3],
                "description": r[4],
                "score": round(score, 3)
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results
    except Exception as e:
        print(f"Search failed: {e}")
        return []


def index_uploaded_image(image_path: str, original_filename: str) -> dict:
    """
    Run CLIP/BLIP to generate description, tags, and embedding, then index it into db.sqlite.
    """
    try:
        init_db()
        # 1. Run classification & captioning
        res = classify_image(image_path)
        if res["status"] != "success":
            raise ValueError(res.get("description", "Classification failed"))
        
        # 2. Generate CLIP embedding
        emb = get_image_embedding(image_path).astype(np.float32)
        emb_bytes = emb.tobytes()
        
        # 3. Create database record
        asset_id = f"up_{uuid.uuid4().hex[:8]}"
        
        # Formulate a clean url path relative to the server
        # The file is stored in backend/static/uploads/
        # So url should be /static/uploads/filename
        url_path = f"/static/uploads/{os.path.basename(image_path)}"
        
        title = original_filename.rsplit('.', 1)[0]
        # Capitalize and clean up title
        title = title.replace('_', ' ').replace('-', ' ').title()
        
        category = res["category"]
        description = res["description"]
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO assets (id, url, title, category, description, embedding) VALUES (?, ?, ?, ?, ?, ?)",
            (asset_id, url_path, title, category, description, emb_bytes)
        )
        conn.commit()
        conn.close()
        
        return {
            "status": "success",
            "id": asset_id,
            "url": url_path,
            "title": title,
            "category": category,
            "description": description,
            "labels": res["labels"]
        }
    except Exception as e:
        return {
            "status": "error",
            "description": f"Failed to index uploaded image: {str(e)}"
        }


def recommend_similar_images(asset_id: str, limit: int = 4) -> list:
    """
    Compute similarity between the selected asset and all other assets in the DB,
    and return the top N matching items.
    """
    try:
        init_db()
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # 1. Retrieve the target embedding
        cursor.execute("SELECT embedding FROM assets WHERE id = ?", (asset_id,))
        row = cursor.fetchone()
        if not row:
            conn.close()
            return []
            
        target_emb = np.frombuffer(row[0], dtype=np.float32)
        
        # 2. Retrieve all other assets
        cursor.execute("SELECT id, url, title, category, description, embedding FROM assets WHERE id != ?", (asset_id,))
        rows = cursor.fetchall()
        conn.close()
        
        results = []
        for r in rows:
            emb = np.frombuffer(r[5], dtype=np.float32)
            dot = np.dot(target_emb, emb)
            norm_t = np.linalg.norm(target_emb)
            norm_e = np.linalg.norm(emb)
            score = float(dot / (norm_t * norm_e)) if (norm_t * norm_e) > 0 else 0.0
            
            results.append({
                "id": r[0],
                "url": r[1],
                "title": r[2],
                "category": r[3],
                "description": r[4],
                "score": round(score, 3)
            })
            
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]
    except Exception as e:
        print(f"Failed to recommend assets: {e}")
        return []

