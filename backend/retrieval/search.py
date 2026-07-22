"""
backend/retrieval/search.py
Owner: Tian Luo — Milestone 1 (semantic search + recommendation)
                  Milestone 4 (hybrid vector + SQL metadata filter for agent)

Responsibilities:
  - semantic_search(query, category_filter, adapter) → ranked results list
  - recommend(asset_id, limit, adapter) → similar assets list
  - sql_metadata_filter(category, keyword) → filtered assets list
  - All three are callable as standalone tools by react_agent.py

Design notes:
  - Cosine similarity is computed in pure NumPy (no vector DB needed at this scale).
  - If an adapter is supplied, embeddings are re-projected before scoring.
  - SQL filter is kept deliberately simple — extend with LIKE / FTS if needed.
"""

import os
import requests
import faiss
import numpy as np
from typing import Optional

from backend.db.database import (
    init_db,
    get_all_assets_with_embeddings,
    get_asset_by_id,
)

ALGO_API_BASE = os.getenv("ALGO_API_BASE", "http://localhost:8001")

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def get_text_embedding_from_algo(text: str) -> np.ndarray:
    resp = requests.post(f"{ALGO_API_BASE}/api/algo/embed/text", json={"text": text}, timeout=10)
    resp.raise_for_status()
    emb = resp.json()["data"]["embedding"]
    return np.array(emb, dtype=np.float32)


def apply_adapter_from_algo(embeddings: np.ndarray, adapter_path: str = "") -> np.ndarray:
    is_1d = (embeddings.ndim == 1)
    if is_1d:
        embeddings_list = [embeddings.tolist()]
    else:
        embeddings_list = embeddings.tolist()
    
    payload = {"embeddings": embeddings_list}
    if adapter_path:
        payload["adapter_path"] = adapter_path
        
    resp = requests.post(f"{ALGO_API_BASE}/api/algo/adapt", json=payload, timeout=20)
    resp.raise_for_status()
    adapted_list = resp.json()["data"]["embeddings"]
    
    res = np.array(adapted_list, dtype=np.float32)
    if is_1d:
        return res[0]
    return res


def get_presigned_url(url: str) -> str:
    import re
    import boto3
    # Match standard S3 HTTP endpoint patterns
    # e.g., https://bucket-name.s3.region-name.amazonaws.com/key-name
    # or https://bucket-name.s3.amazonaws.com/key-name
    match = re.match(r"https?://([^.]+)\.s3[^/]*\.amazonaws\.com/(.+)", url)
    if not match:
        return url
    
    import urllib.parse
    bucket = match.group(1)
    key = urllib.parse.unquote(match.group(2))
    
    try:
        s3_region = os.getenv("UKAHT_S3_REGION")
        s3_client = boto3.client("s3", region_name=s3_region) if s3_region else boto3.client("s3")
        presigned_url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket, 'Key': key},
            ExpiresIn=3600
        )
        return presigned_url
    except Exception as e:
        print(f"[WARN] S3 pre-signing failed for {url}: {e}")
        return url


def _row_to_result(row: dict, score: float) -> dict:
    url = row["url"]
    if url.startswith("http") and (".s3." in url or "s3.amazonaws.com" in url):
        url = get_presigned_url(url)
    return {
        "id": row["id"],
        "url": url,
        "title": row["title"],
        "category": row["category"],
        "description": row["description"],
        "score": round(score, 4),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def semantic_search(
    query: str,
    category_filter: Optional[str] = None,
    adapter=None,
) -> list[dict]:
    """
    Encode a text query with CLIP, optionally re-project with adapter,
    then score all assets using FAISS similarity search.

    Args:
        query:           Natural-language search string.
        category_filter: If provided, only assets with a matching category
                         are scored (case-insensitive).
        adapter:         Optional AdapterModel or path (str) — pass None to use vanilla CLIP.

    Returns:
        List of asset dicts sorted by descending score.
    """
    init_db()

    rows = get_all_assets_with_embeddings()
    
    # Filter rows by category first
    matched_rows = []
    for r in rows:
        if r["embedding"] is None:
            continue
        # Safe category comparison in case category is None/NULL in database
        asset_cat = r["category"] or ""
        if category_filter and asset_cat.lower() != category_filter.lower():
            continue
        matched_rows.append(r)

    if not matched_rows:
        return []

    # Prepare database embeddings
    raw_embeddings = []
    for r in matched_rows:
        emb = np.frombuffer(r["embedding"], dtype=np.float32)
        raw_embeddings.append(emb)

    embeddings_matrix = np.vstack(raw_embeddings).astype("float32")
    if adapter:
        adapter_path = adapter if isinstance(adapter, str) else ""
        embeddings_matrix = apply_adapter_from_algo(embeddings_matrix, adapter_path)

    if not query.strip():
        scores_arr = [1.0] * len(matched_rows)
        indices_arr = list(range(len(matched_rows)))
    else:
        # Retrieve text embedding (keep in CLIP text space; image embeddings are adapted)
        query_emb = get_text_embedding_from_algo(query)
        query_emb = query_emb.astype("float32").reshape(1, -1)

        # Build the FAISS Index
        # Since query and asset embeddings are L2-normalized, IndexFlatIP is equivalent to cosine similarity.
        index = faiss.IndexFlatIP(512)
        index.add(embeddings_matrix)

        # Query FAISS index
        k = len(matched_rows)
        scores_res, indices_res = index.search(query_emb, k)
        scores_arr = scores_res[0]
        indices_arr = indices_res[0]

    # Reconstruct sorted result list with visual stacking (grouping highly similar items)
    results = []
    primary_embeddings = []
    primary_indices = []
    
    # We define a threshold for high visual similarity, e.g. 0.85
    SIMILARITY_THRESHOLD = 0.85

    for score, idx in zip(scores_arr, indices_arr):
        if idx == -1:
            continue
        row = matched_rows[idx]
        emb = embeddings_matrix[idx]
        
        is_stacked = False
        for p_idx, p_emb in zip(primary_indices, primary_embeddings):
            sim = float(np.dot(emb, p_emb))
            if sim >= SIMILARITY_THRESHOLD:
                if "stacked_assets" not in results[p_idx]:
                    results[p_idx]["stacked_assets"] = []
                results[p_idx]["stacked_assets"].append(_row_to_result(row, float(score)))
                is_stacked = True
                break
                
        if not is_stacked:
            res = _row_to_result(row, float(score))
            res["stacked_assets"] = []
            results.append(res)
            primary_embeddings.append(emb)
            primary_indices.append(len(results) - 1)

    return results


def recommend(
    asset_id: str,
    limit: int = 6,
    adapter=None,
) -> list[dict]:
    """
    Find the most visually similar assets to a given asset using FAISS similarity search.

    Args:
        asset_id: ID of the reference asset already in the database.
        limit:    Maximum number of results to return.
        adapter:  Optional AdapterModel or path (str).

    Returns:
        List of similar asset dicts sorted by descending score.
    """
    init_db()

    target = get_asset_by_id(asset_id)
    if not target or not target["embedding"]:
        return []

    target_emb = np.frombuffer(target["embedding"], dtype=np.float32)
    if adapter:
        adapter_path = adapter if isinstance(adapter, str) else ""
        target_emb = apply_adapter_from_algo(target_emb, adapter_path)
    target_emb = target_emb.astype("float32").reshape(1, -1)

    rows = get_all_assets_with_embeddings()
    matched_rows = []
    for r in rows:
        if r["id"] == asset_id or r["embedding"] is None:
            continue
        matched_rows.append(r)

    if not matched_rows:
        return []

    # Build FAISS index for matched assets
    index = faiss.IndexFlatIP(512)
    raw_embeddings = []
    for r in matched_rows:
        emb = np.frombuffer(r["embedding"], dtype=np.float32)
        raw_embeddings.append(emb)

    embeddings_matrix = np.vstack(raw_embeddings).astype("float32")
    if adapter:
        adapter_path = adapter if isinstance(adapter, str) else ""
        embeddings_matrix = apply_adapter_from_algo(embeddings_matrix, adapter_path)

    index.add(embeddings_matrix)

    # Query FAISS index for top visually similar assets
    k = min(len(matched_rows), limit)
    scores, indices = index.search(target_emb, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        row = matched_rows[idx]
        results.append(_row_to_result(row, float(score)))

    return results


def sql_metadata_filter(
    category: Optional[str] = None,
    keyword: Optional[str] = None,
    base_code: Optional[str] = None,
    subject_type: Optional[str] = None,
    shooting_year: Optional[str] = None,
    copyright: Optional[str] = None,
    data_source: Optional[str] = None,
) -> list[dict]:
    """
    Pure SQL metadata filter — used by the LLM agent when it detects
    a structured query (e.g. 'show me all Equipment photos').

    Args:
        category:      Exact category string to match (case-insensitive).
        keyword:       Substring to search in title or description.
        base_code:     Exact base code (e.g., 'E', 'W', 'A').
        subject_type:  Substring or exact subject type.
        shooting_year: Exact year or season string.
        copyright:     Copyright/credit holder substring or exact match.
        data_source:   Data source ('original' or 'new_addition').

    Returns:
        List of matching asset dicts (no score field).
    """
    from backend.db.database import get_connection

    init_db()
    conditions = []
    params = []

    if category:
        conditions.append("LOWER(category) = LOWER(?)")
        params.append(category)
    if keyword:
        conditions.append("(title LIKE ? OR description LIKE ?)")
        params.extend([f"%{keyword}%", f"%{keyword}%"])
    if base_code:
        conditions.append("LOWER(base_code) = LOWER(?)")
        params.append(base_code)
    if subject_type:
        conditions.append("LOWER(subject_type) LIKE LOWER(?)")
        params.append(f"%{subject_type}%")
    if shooting_year:
        normalized_year = shooting_year.replace("-", "_")
        conditions.append("(shooting_year = ? OR shooting_year = ?)")
        params.extend([shooting_year, normalized_year])
    if copyright:
        conditions.append("LOWER(copyright) LIKE LOWER(?)")
        params.append(f"%{copyright}%")
    if data_source:
        conditions.append("LOWER(data_source) = LOWER(?)")
        params.append(data_source)

    where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    sql = f"SELECT id, url, title, category, description, base_code, subject_type, shooting_year, copyright, data_source FROM assets {where_clause}"

    with get_connection() as conn:
        rows = conn.execute(sql, params).fetchall()

    results = []
    for r in rows:
        d = dict(r)
        url = d["url"]
        if url.startswith("http") and (".s3." in url or "s3.amazonaws.com" in url):
            d["url"] = get_presigned_url(url)
        results.append(d)
    return results
