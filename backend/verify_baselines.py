import os
import sys

# Ensure backend folder is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.functions import (
    classify_image,
    semantic_search,
    index_uploaded_image,
    recommend_similar_images
)

def run_tests():
    print("=== Testing UKAHT AI Baselines ===")
    
    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sample_img = os.path.join(base_dir, "static", "images", "mock_sample1.png")
    
    # 1. Classification
    print("\n--- 1. Testing Image Classification (CLIP + BLIP) ---")
    if not os.path.exists(sample_img):
        print(f"Error: Sample image not found at {sample_img}")
        return
        
    print(f"Running classify_image on {sample_img}...")
    res = classify_image(sample_img)
    print("Result:", res)
    
    # 2. Semantic Search
    print("\n--- 2. Testing Semantic Search (MiniLM + SQLite) ---")
    query = "house"
    print(f"Running semantic_search with query '{query}'...")
    res = semantic_search(query)
    print("Found items:")
    for item in res:
         print(f"  - Title: {item['title']}, Score: {item['score']}")
         

    # 5. Image Indexing and Recommendations
    print("\n--- 5. Testing Image Indexing & Recommendation Pipelines ---")
    test_upload_dest = os.path.join(base_dir, "static", "uploads", "test_verify_upload.png")
    os.makedirs(os.path.dirname(test_upload_dest), exist_ok=True)
    import shutil
    shutil.copy(sample_img, test_upload_dest)
    
    print(f"Indexing uploaded image {test_upload_dest}...")
    idx_res = index_uploaded_image(test_upload_dest, "test_verify_upload.png")
    print("Indexing Result:", idx_res)
    
    if idx_res["status"] == "success":
        new_id = idx_res["id"]
        print(f"Running recommend_similar_images for asset ID '{new_id}'...")
        rec_res = recommend_similar_images(new_id, limit=3)
        print("Recommendations:")
        for rec in rec_res:
            print(f"  - Title: {rec['title']}, Category: {rec['category']}, Score: {rec['score']}")
            
    print("\n=== Baselines Verification Complete ===")

if __name__ == "__main__":
    run_tests()
