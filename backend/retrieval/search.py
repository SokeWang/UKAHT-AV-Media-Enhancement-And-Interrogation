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
    try:
        from algorithm.models.clip_model import get_text_embedding
        return get_text_embedding(text)
    except Exception as exc:
        print(f"[SEARCH WARN] Local text embedding failed, falling back to HTTP: {exc}")
        resp = requests.post(f"{ALGO_API_BASE}/api/algo/embed/text", json={"text": text}, timeout=15)
        resp.raise_for_status()
        emb = resp.json()["data"]["embedding"]
        return np.array(emb, dtype=np.float32)


def apply_adapter_from_algo(embeddings: np.ndarray, adapter_path: str = "") -> np.ndarray:
    try:
        from algorithm.models.adapter import apply_adapter
        return apply_adapter(embeddings, adapter_path=adapter_path)
    except Exception as exc:
        print(f"[SEARCH WARN] Local adapter execution failed, falling back to HTTP: {exc}")
        is_1d = (embeddings.ndim == 1)
        if is_1d:
            embeddings_list = [embeddings.tolist()]
        else:
            embeddings_list = embeddings.tolist()
        
        payload = {"embeddings": embeddings_list}


def apply_text_adapter_from_algo(embeddings: np.ndarray, adapter_path: str = "") -> np.ndarray:
    try:
        from algorithm.models.adapter import apply_text_adapter, load_adapter
        adapter = load_adapter(adapter_path)
        return apply_text_adapter(embeddings, adapter)
    except Exception as exc:
        return embeddings
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
    if not url:
        return ""
    import urllib.parse

    # Return proxy endpoint so image content is fetched server-side from S3 and streamed directly to browser
    if url.startswith("http") or url.startswith("s3://"):
        encoded_url = urllib.parse.quote(url, safe="")
        return f"/api/image-proxy?url={encoded_url}"
    
    encoded_url = urllib.parse.quote(url, safe="")
    return f"/api/image-proxy?url={encoded_url}"


def _row_to_result(row: dict, score: float) -> dict:
    url = row["url"]
    if url.startswith("http") and (".s3." in url or "s3.amazonaws.com" in url):
        url = get_presigned_url(url)
    clean_cat = row.get("cluster_label") or row.get("category") or "Antarctic Heritage"
    
    title = row.get("title") or "Untitled"
    base = row.get("base_code") or "N/A"
    year = row.get("shooting_year") or "Unknown Year"
    sub_type = row.get("subject_type") or "N/A"
    credit = row.get("copyright") or "UKAHT Collection"
    raw_desc = row.get("description") or ""

    # Rich composite concatenation of ALL metadata fields for Semantic Search & RAG
    composite_desc = (
        f"[Title: {title}] | [Base: Base {base}] | [Shooting Year: {year}] | "
        f"[Category: {clean_cat}] | [Subject Type: {sub_type}] | [Credit: {credit}] | "
        f"[Visual Content: {raw_desc}]"
    )

    return {
        "id": row["id"],
        "url": url,
        "title": title,
        "category": clean_cat,
        "cluster_label": clean_cat,
        "base_code": base,
        "subject_type": sub_type,
        "shooting_year": year,
        "copyright": credit,
        "description": composite_desc,
        "raw_caption": raw_desc,
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
        # Retrieve text embedding and adapt via text branch if dual adapter is active
        query_emb = get_text_embedding_from_algo(query)
        query_emb = query_emb.astype("float32").reshape(1, -1)
        if adapter:
            adapter_path = adapter if isinstance(adapter, str) else ""
            query_emb = apply_text_adapter_from_algo(query_emb, adapter_path)

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


def execute_sql_query(where_clause: Optional[str] = None, sql: Optional[str] = None) -> list[dict]:
    """
    Direct Text-to-SQL execution engine:
    Allows AI Agent to generate raw PostgreSQL WHERE clauses or SELECT statements dynamically.
    Includes safety validation to enforce read-only access.
    """
    from backend.db.database import get_connection, RealDictCursor
    import re

    init_db()

    if sql and sql.strip():
        clean_sql = sql.strip()
        # Security check: must be read-only SELECT
        if not re.match(r'(?i)^\s*SELECT\b', clean_sql) or any(w in clean_sql.upper() for w in ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'TRUNCATE']):
            raise ValueError("Only read-only SELECT queries are permitted.")
        query = clean_sql
    elif where_clause and where_clause.strip():
        clean_where = where_clause.strip()
        if any(w in clean_where.upper() for w in ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'TRUNCATE', ';']):
            raise ValueError("Unsafe SQL keywords detected in WHERE clause.")
        if not clean_where.upper().startswith("WHERE"):
            clean_where = f"WHERE {clean_where}"
        query = f"SELECT id, url, title, category, description, cluster_label, base_code, subject_type, shooting_year, copyright, data_source FROM assets {clean_where} LIMIT 100"
    else:
        query = "SELECT id, url, title, category, description, cluster_label, base_code, subject_type, shooting_year, copyright, data_source FROM assets ORDER BY shooting_year ASC LIMIT 100"

    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()

    results = []
    for r in rows:
        d = dict(r)
        url = d.get("url", "")
        if url.startswith("http") and (".s3." in url or "s3.amazonaws.com" in url):
            d["url"] = get_presigned_url(url)
        d["category"] = d.get("cluster_label") or d.get("category") or "Antarctic Heritage"
        results.append(d)
    return results


def sql_metadata_filter(
    keyword: Optional[str] = None,
    **filters
) -> list[dict]:
    """
    Fallback Metadata Filter — converts filter dictionary to dynamic SQL WHERE clause.
    """
    where_parts = []
    for k, v in filters.items():
        if v and str(v).strip():
            val = str(v).strip()
            # Clean natural language prefixes like 'Base E' -> 'E'
            val = re.sub(r'(?i)^(base|station)\s+', '', val)
            if re.search(r'\d{4}', val) and ('<' in val or '>' in val):
                m = re.search(r'([<>=]+)\s*(\d{4})', val)
                if m:
                    where_parts.append(f"{k} {m.group(1)} '{m.group(2)}'")
            elif re.search(r'\d{4}', val):
                m = re.search(r'\d{4}', val)
                where_parts.append(f"{k} ILIKE '%%{m.group(0)}%%'")
            else:
                where_parts.append(f"{k} ILIKE '%%{val}%%'")

    if keyword and str(keyword).strip():
        kw = str(keyword).strip()
        where_parts.append(f"(title ILIKE '%%{kw}%%' OR description ILIKE '%%{kw}%%' OR category ILIKE '%%{kw}%%')")

    where_clause = " AND ".join(where_parts) if where_parts else None
    return execute_sql_query(where_clause=where_clause)
