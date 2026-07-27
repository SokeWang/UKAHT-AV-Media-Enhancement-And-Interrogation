"""
backend/evaluation/cluster_assets.py

Runs pure numpy K-Means clustering on asset embeddings to generate clean, high-level semantic labels
(e.g., 'Exterior Heritage & Huts', 'Artifacts & Museum Display', 'Interior Living Spaces',
'Polar Landscape & Glaciers', 'Expedition Equipment & Vessels').

Updates PostgreSQL database table `assets` with `cluster_label` and cleans up noisy `category` fields.
"""

import os
import numpy as np
import psycopg2
from backend.db.database import get_connection

CANDIDATE_CLUSTERS = [
    {"label": "Exterior Heritage & Huts", "query": "exterior view of antarctic wooden huts and historic buildings"},
    {"label": "Artifacts & Museum Display", "query": "museum artifacts tools cans equipment and historical items"},
    {"label": "Interior Living Spaces", "query": "interior rooms living quarters furniture and indoor spaces"},
    {"label": "Polar Landscape & Glaciers", "query": "polar glaciers icebergs snow mountains and antarctic landscape"},
    {"label": "Expedition Equipment & Vessels", "query": "expedition ships vehicles sledges and outdoor equipment"},
]


def kmeans_numpy(X: np.ndarray, k: int = 5, max_iter: int = 25):
    """Pure numpy implementation of K-Means clustering."""
    np.random.seed(42)
    n_samples = X.shape[0]
    init_indices = np.random.choice(n_samples, k, replace=False)
    centers = X[init_indices].copy()
    labels = np.zeros(n_samples, dtype=int)

    for iteration in range(max_iter):
        # Compute euclidean distance between samples and cluster centers
        # dists shape: (n_samples, k)
        dists = np.linalg.norm(X[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
        new_labels = np.argmin(dists, axis=1)

        if np.array_equal(labels, new_labels):
            break
        labels = new_labels

        # Update centers
        new_centers = np.zeros_like(centers)
        for i in range(k):
            members = X[labels == i]
            if len(members) > 0:
                new_centers[i] = members.mean(axis=0)
            else:
                new_centers[i] = centers[i]
        centers = new_centers

    return labels, centers


def run_clustering(n_clusters: int = 5):
    print("=== STARTING ASSET EMBEDDING CLUSTERING ===")

    # 1. Fetch embeddings from Postgres
    with get_connection() as conn:
        with conn.cursor() as cursor:
            # Ensure cluster_label column exists
            cursor.execute("ALTER TABLE assets ADD COLUMN IF NOT EXISTS cluster_label TEXT;")
            conn.commit()

            cursor.execute("SELECT id, title, category, subject_type, embedding FROM assets WHERE embedding IS NOT NULL;")
            rows = cursor.fetchall()

    if not rows:
        print("[WARN] No assets with embeddings found in database!")
        return

    print(f"Loaded {len(rows)} asset embeddings from PostgreSQL database.")

    asset_ids = []
    embeddings_list = []
    titles = []

    for r in rows:
        a_id, title, cat, sub_type, emb_bytes = r
        if not emb_bytes:
            continue
        emb = np.frombuffer(emb_bytes, dtype=np.float32)
        if len(emb) == 512:
            asset_ids.append(a_id)
            embeddings_list.append(emb)
            titles.append(title or "")

    if not embeddings_list:
        print("[ERROR] Failed to unpack float32 embeddings.")
        return

    X = np.vstack(embeddings_list).astype("float32")
    # Normalize for cosine similarity clustering
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X_norm = X / norms

    print(f"Running Numpy K-Means (K={n_clusters}) on normalized matrix shape: {X_norm.shape}...")
    cluster_indices, cluster_centers = kmeans_numpy(X_norm, k=n_clusters)

    # 2. Get text embeddings for candidate cluster names using algorithm service or keyword heuristic
    import requests
    algo_url = os.getenv("ALGO_API_BASE", "http://localhost:8001")

    cluster_names = {}
    for k in range(n_clusters):
        k_indices = np.where(cluster_indices == k)[0]
        sample_titles = [titles[i] for i in k_indices if titles[i]][:15]
        combined_titles = " ".join(sample_titles).lower()

        if any(w in combined_titles for w in ["artifact", "can", "boot", "tin", "box", "tool", "museum", "shoe", "bottle"]):
            assigned_name = "Artifacts & Museum Display"
        elif any(w in combined_titles for w in ["interior", "room", "bed", "kitchen", "inside", "bunk", "desk"]):
            assigned_name = "Interior Living Spaces"
        elif any(w in combined_titles for w in ["glacier", "ice", "mountain", "bay", "sea", "snow", "landscape"]):
            assigned_name = "Polar Landscape & Glaciers"
        elif any(w in combined_titles for w in ["ship", "boat", "sledge", "plane", "vehicle", "tractor"]):
            assigned_name = "Expedition Equipment & Vessels"
        else:
            # Fallback zero-shot text matching via algorithm service if reachable
            assigned_name = CANDIDATE_CLUSTERS[k % len(CANDIDATE_CLUSTERS)]["label"]
            try:
                best_label = assigned_name
                best_sim = -1.0
                center = cluster_centers[k]
                center_norm = center / (np.linalg.norm(center) + 1e-9)

                for candidate in CANDIDATE_CLUSTERS:
                    resp = requests.post(f"{algo_url}/api/text-embedding", json={"text": candidate["query"]}, timeout=2)
                    if resp.status_code == 200:
                        txt_emb = np.array(resp.json().get("data", {}).get("embedding", []), dtype=np.float32)
                        if len(txt_emb) == 512:
                            txt_norm = txt_emb / (np.linalg.norm(txt_emb) + 1e-9)
                            sim = float(np.dot(center_norm, txt_norm))
                            if sim > best_sim:
                                best_sim = sim
                                best_label = candidate["label"]
                assigned_name = best_label
            except Exception:
                pass

        cluster_names[k] = assigned_name
        print(f"  Cluster {k} ({len(k_indices)} items) -> Assigned Label: \"{assigned_name}\"")

    # 3. Update database with clean cluster labels
    print("Writing clean cluster labels to PostgreSQL assets table...")
    with get_connection() as conn:
        with conn.cursor() as cursor:
            for a_id, k in zip(asset_ids, cluster_indices):
                label = cluster_names[k]
                cursor.execute(
                    "UPDATE assets SET cluster_label = %s, category = %s WHERE id = %s;",
                    (label, label, a_id)
                )
            conn.commit()

    print("=== ASSET EMBEDDING CLUSTERING COMPLETED SUCCESSFULLY ===")


if __name__ == "__main__":
    run_clustering()
