"""
backend/db/database.py
Owner: Tian Luo — Milestone 1 (SQLite schema) + Milestone 2 (annotation updates)
Refactored for PostgreSQL container (Option A)

Responsibilities:
  - Define and initialise the tables schema in PostgreSQL
  - Provide a thread-safe connection context manager using psycopg2
  - Expose low-level CRUD helpers used by retrieval/search.py and ingest/
"""

import os
import hashlib
import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager
from typing import Optional

# ---------------------------------------------------------------------------
# Schema Definitions
# ---------------------------------------------------------------------------
_CREATE_ASSETS_TABLE = """
CREATE TABLE IF NOT EXISTS assets (
    id            TEXT PRIMARY KEY,
    url           TEXT NOT NULL,
    title         TEXT,
    category      TEXT,
    description   TEXT,      -- auto-generated or manually corrected caption
    embedding     BYTEA,     -- CLIP 512-dim float32 vector (BYTEA in Postgres)
    base_code     TEXT,      -- 'A', 'E', 'W'
    subject_type  TEXT,      -- 'Exterior', 'Main Hut', 'Artifact', 'SfM'
    shooting_year TEXT,      -- '1958', '2011_12', '2025'
    copyright     TEXT,      -- credit/copyright holder
    data_source   TEXT       -- 'original' or 'new_addition'
)
"""

_CREATE_GOLDEN_TABLE = """
CREATE TABLE IF NOT EXISTS golden_test_set (
    id          TEXT PRIMARY KEY,
    asset_id    TEXT NOT NULL,
    caption     TEXT NOT NULL,    -- manually verified caption
    annotator   TEXT,
    created_at  TEXT DEFAULT (to_char(now() AT TIME ZONE 'UTC', 'YYYY-MM-DD HH24:MI:SS'))
)
"""

_CREATE_USERS_TABLE = """
CREATE TABLE IF NOT EXISTS users (
    username       TEXT PRIMARY KEY,
    password_hash  TEXT NOT NULL
)
"""


# ---------------------------------------------------------------------------
# Connection helper
# ---------------------------------------------------------------------------
@contextmanager
def get_connection():
    """Context manager that yields a psycopg2 Connection and commits/rolls back."""
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    database = os.getenv("POSTGRES_DB", "ukaht")
    user = os.getenv("POSTGRES_USER", "postgres")
    password = os.getenv("POSTGRES_PASSWORD", "postgres")
    
    conn = psycopg2.connect(
        host=host,
        port=port,
        database=database,
        user=user,
        password=password
    )
    try:
        yield conn
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------
def init_db() -> None:
    """Create tables if they do not yet exist. Safe to call repeatedly."""
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(_CREATE_ASSETS_TABLE)
            cursor.execute(_CREATE_GOLDEN_TABLE)
            cursor.execute(_CREATE_USERS_TABLE)
            
            # Schema migration check: dynamically add columns if they do not exist
            cursor.execute(
                """SELECT column_name FROM information_schema.columns 
                   WHERE table_name = 'assets'"""
            )
            columns = [row["column_name"] for row in cursor.fetchall()]
            new_columns = {
                "base_code": "TEXT",
                "subject_type": "TEXT",
                "shooting_year": "TEXT",
                "copyright": "TEXT",
                "data_source": "TEXT"
            }
            for col_name, col_type in new_columns.items():
                if col_name not in columns:
                    cursor.execute(f"ALTER TABLE assets ADD COLUMN {col_name} {col_type}")
                    
            # Seed default user if empty
            cursor.execute("SELECT COUNT(*) as count FROM users")
            row = cursor.fetchone()
            if row["count"] == 0:
                default_pwd_hash = hashlib.sha256("ukaht2026".encode("utf-8")).hexdigest()
                cursor.execute(
                    "INSERT INTO users (username, password_hash) VALUES (%s, %s)",
                    ("admin", default_pwd_hash)
                )
                
        conn.commit()


# ---------------------------------------------------------------------------
# CRUD helpers
# ---------------------------------------------------------------------------

def insert_asset(asset_id: str, url: str, title: str, category: str,
                 description: str, embedding_bytes: bytes,
                 base_code: Optional[str] = None,
                 subject_type: Optional[str] = None,
                 shooting_year: Optional[str] = None,
                 copyright: Optional[str] = None,
                 data_source: Optional[str] = None) -> None:
    """Insert or replace a single asset record."""
    emb_param = psycopg2.Binary(embedding_bytes) if embedding_bytes is not None else None
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """INSERT INTO assets
                   (id, url, title, category, description, embedding,
                    base_code, subject_type, shooting_year, copyright, data_source)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT (id) DO UPDATE SET
                       url = EXCLUDED.url,
                       title = EXCLUDED.title,
                       category = EXCLUDED.category,
                       description = EXCLUDED.description,
                       embedding = EXCLUDED.embedding,
                       base_code = EXCLUDED.base_code,
                       subject_type = EXCLUDED.subject_type,
                       shooting_year = EXCLUDED.shooting_year,
                       copyright = EXCLUDED.copyright,
                       data_source = EXCLUDED.data_source""",
                (asset_id, url, title, category, description, emb_param,
                 base_code, subject_type, shooting_year, copyright, data_source)
            )
        conn.commit()


def update_asset_description(asset_id: str, description: str) -> None:
    """Update the caption/description for an existing asset (annotation UI)."""
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "UPDATE assets SET description = %s WHERE id = %s",
                (description, asset_id)
            )
        conn.commit()


def get_all_assets() -> list[dict]:
    """Return all assets as a list of dicts (without embedding bytes)."""
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                "SELECT id, url, title, category, description, cluster_label, base_code, subject_type, shooting_year, copyright, data_source FROM assets"
            )
            rows = cursor.fetchall()
    return [dict(r) for r in rows]


def get_asset_by_id(asset_id: str) -> Optional[dict]:
    """Return a single asset dict including embedding bytes, or None."""
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                "SELECT id, url, title, category, description, cluster_label, embedding, base_code, subject_type, shooting_year, copyright, data_source FROM assets WHERE id = %s",
                (asset_id,)
            )
            row = cursor.fetchone()
    if row:
        res = dict(row)
        if res.get("embedding"):
            res["embedding"] = bytes(res["embedding"])
        return res
    return None


def get_all_assets_with_embeddings() -> list[dict]:
    """Return all assets including embedding blobs — used by search."""
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                "SELECT id, url, title, category, description, cluster_label, embedding, base_code, subject_type, shooting_year, copyright, data_source FROM assets"
            )
            rows = cursor.fetchall()
    res_list = []
    for r in rows:
        d = dict(r)
        if d.get("embedding"):
            d["embedding"] = bytes(d["embedding"])
        res_list.append(d)
    return res_list


def insert_golden_entry(entry_id: str, asset_id: str,
                        caption: str, annotator: str = "") -> None:
    """Save a manually verified golden test-set entry."""
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """INSERT INTO golden_test_set
                   (id, asset_id, caption, annotator)
                   VALUES (%s, %s, %s, %s)
                   ON CONFLICT (id) DO UPDATE SET
                       asset_id = EXCLUDED.asset_id,
                       caption = EXCLUDED.caption,
                       annotator = EXCLUDED.annotator""",
                (entry_id, asset_id, caption, annotator)
            )
        conn.commit()


def get_golden_test_set() -> list[dict]:
    """Return all golden test-set entries."""
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                "SELECT id, asset_id, caption, annotator, created_at FROM golden_test_set"
            )
            rows = cursor.fetchall()
    return [dict(r) for r in rows]


def check_duplicate_image(new_emb_bytes: bytes, threshold: float = 0.99) -> Optional[str]:
    """
    Check if a visually duplicate image already exists in the database.
    Returns the asset ID of the duplicate if found, otherwise None.
    """
    import numpy as np
    if not new_emb_bytes:
        return None
    new_emb = np.frombuffer(new_emb_bytes, dtype=np.float32)
    norm = np.linalg.norm(new_emb)
    if norm > 0:
        new_emb = new_emb / norm
        
    all_assets = get_all_assets_with_embeddings()
    for asset in all_assets:
        if not asset["embedding"]:
            continue
        emb = np.frombuffer(asset["embedding"], dtype=np.float32)
        emb_norm = np.linalg.norm(emb)
        if emb_norm > 0:
            emb = emb / emb_norm
        similarity = np.dot(new_emb, emb)
        if similarity >= threshold:
            return asset["id"]
    return None


def verify_user(username: str, password_plain: str) -> bool:
    """Verify user password hash from database."""
    pwd_hash = hashlib.sha256(password_plain.encode("utf-8")).hexdigest()
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                "SELECT password_hash FROM users WHERE username = %s",
                (username,)
            )
            row = cursor.fetchone()
    if row and row["password_hash"] == pwd_hash:
        return True
    return False
