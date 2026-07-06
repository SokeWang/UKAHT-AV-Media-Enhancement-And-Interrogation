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

import faiss
import numpy as np
from typing import Optional

from backend.db.database import (
    init_db,
    get_all_assets_with_embeddings,
    get_asset_by_id,
)
from backend.models.clip_model import get_text_embedding, get_image_embedding
from backend.models.adapter import apply_adapter

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _row_to_result(row: dict, score: float) -> dict:
    return {
        "id": row["id"],
        "url": row["url"],
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
        adapter:         Optional AdapterModel — pass None to use vanilla CLIP.

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
        if category_filter and r["category"].lower() != category_filter.lower():
            continue
        matched_rows.append(r)

    if not matched_rows:
        return []

    if not query.strip():
        # Return all matched assets with score 1.0 when query is empty
        return [_row_to_result(r, 1.0) for r in matched_rows]

    # Retrieve and adapt query embedding
    query_emb = get_text_embedding(query)
    query_emb = apply_adapter(query_emb, adapter)
    query_emb = query_emb.astype("float32").reshape(1, -1)

    # Build the FAISS Index
    # Since query and asset embeddings are L2-normalized, IndexFlatIP is equivalent to cosine similarity.
    index = faiss.IndexFlatIP(512)

    # Prepare database embeddings
    embeddings = []
    for r in matched_rows:
        emb = np.frombuffer(r["embedding"], dtype=np.float32)
        emb = apply_adapter(emb, adapter)
        embeddings.append(emb)

    embeddings_matrix = np.vstack(embeddings).astype("float32")
    index.add(embeddings_matrix)

    # Query FAISS index
    k = len(matched_rows)
    scores, indices = index.search(query_emb, k)

    # Reconstruct sorted result list
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        row = matched_rows[idx]
        results.append(_row_to_result(row, float(score)))

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
        adapter:  Optional AdapterModel.

    Returns:
        List of similar asset dicts sorted by descending score.
    """
    init_db()

    target = get_asset_by_id(asset_id)
    if not target or not target["embedding"]:
        return []

    target_emb = np.frombuffer(target["embedding"], dtype=np.float32)
    target_emb = apply_adapter(target_emb, adapter)
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
    embeddings = []
    for r in matched_rows:
        emb = np.frombuffer(r["embedding"], dtype=np.float32)
        emb = apply_adapter(emb, adapter)
        embeddings.append(emb)

    embeddings_matrix = np.vstack(embeddings).astype("float32")
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
) -> list[dict]:
    """
    Pure SQL metadata filter — used by the LLM agent when it detects
    a structured query (e.g. 'show me all Equipment photos').

    Args:
        category: Exact category string to match (case-insensitive).
        keyword:  Substring to search in title or description.

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

    where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    sql = f"SELECT id, url, title, category, description FROM assets {where_clause}"

    with get_connection() as conn:
        rows = conn.execute(sql, params).fetchall()

    return [dict(r) for r in rows]
