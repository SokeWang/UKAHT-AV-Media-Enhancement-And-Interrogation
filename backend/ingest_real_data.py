import os
import sys
import glob
import sqlite3
import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.functions import DB_PATH

def ingest():
    print("=== UKAHT Real Dataset Ingestion (Baseline Indexing) ===")
    
    dataset_root = "/Users/wangpeidong/UKAHT-Project/UK Antarctic Heritage Trust Data"
    if not os.path.exists(dataset_root):
        print(f"Error: Dataset path not found: {dataset_root}")
        return
        
    # Find images
    image_patterns = [
        "SfM/SfM_A_external/*.JPG",
        "Condition_Artefacts/Artefacts_A_1_h_16_Ionospherics/A_1_h_16/*.jpg",
        "Condition_Artefacts/Artefacts_A_1_h_16_Ionospherics/A_1_h_16/*.JPG",
        "Various_A/*.jpg",
        "Various_A/*.JPG",
    ]
    
    matched_images = []
    for pattern in image_patterns:
        full_pattern = os.path.join(dataset_root, pattern)
        files = glob.glob(full_pattern)
        print(f"Pattern {pattern} matched {len(files)} files.")
        matched_images.extend(files)
        
    if not matched_images:
        print("No images found in dataset folders.")
        return
        
    # Sample up to 15 images to build the baseline index quickly
    sampled_images = []
    folders = set(os.path.dirname(p) for p in matched_images)
    for folder in folders:
        folder_imgs = [p for p in matched_images if os.path.dirname(p) == folder]
        sampled_images.extend(folder_imgs[:4]) # take up to 4 from each subfolder
        
    print(f"Sampled {len(sampled_images)} images for embedding indexing.")
    
    # Initialize SQLite and clean assets table
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS assets")
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
    
    from backend.functions import (
        get_image_embeddings_batch, 
        get_blip_captions_batch, 
        classify_images_batch
    )
    
    candidate_labels = ["building", "landscape", "artifact", "equipment", "wildlife"]
    batch_size = 4
    
    for i in range(0, len(sampled_images), batch_size):
        batch_paths = sampled_images[i:i + batch_size]
        print(f"\nProcessing batch {i // batch_size + 1} ({len(batch_paths)} images)...")
        
        try:
            pil_imgs = [Image.open(p).convert("RGB") for p in batch_paths]
            
            embs = get_image_embeddings_batch(pil_imgs)
            captions = get_blip_captions_batch(pil_imgs)
            classifications = classify_images_batch(pil_imgs, candidate_labels)
            
            for idx_in_batch, img_path in enumerate(batch_paths):
                global_idx = i + idx_in_batch
                filename = os.path.basename(img_path)
                rel_path = os.path.relpath(img_path, dataset_root)
                url_path = f"/data/{rel_path}"
                
                emb_bytes = embs[idx_in_batch].astype(np.float32).tobytes()
                category = classifications[idx_in_batch]["category"]
                description = captions[idx_in_batch]
                title = f"{category} Asset ({filename})"
                
                cursor.execute(
                    "INSERT INTO assets (id, url, title, category, description, embedding) VALUES (?, ?, ?, ?, ?, ?)",
                    (f"real_{global_idx}", url_path, title, category, description, emb_bytes)
                )
            
            conn.commit()
            print(f"  -> Successfully indexed batch of size {len(batch_paths)}")
        except Exception as e:
            print(f"  -> Error processing batch starting at index {i}: {e}")
            
    conn.close()
    print("=== Ingestion Completed Successfully ===")

if __name__ == "__main__":
    ingest()
