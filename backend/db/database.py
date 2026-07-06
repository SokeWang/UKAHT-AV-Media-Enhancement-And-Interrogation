"""
backend/db/database.py
Owner: Tian Luo — Milestone 1 (SQLite schema) + Milestone 2 (annotation updates)

Responsibilities:
  - Define and initialise the 'assets' table schema
  - Provide a thread-safe connection context manager
  - Expose low-level CRUD helpers used by retrieval/search.py and ingest/
"""

import os
import sqlite3
from contextlib import contextmanager
from typing import Optional

# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------
_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "db.sqlite")
DB_PATH = os.path.normpath(_DB_PATH)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
_CREATE_ASSETS_TABLE = """
CREATE TABLE IF NOT EXISTS assets (
    id          TEXT PRIMARY KEY,
    url         TEXT NOT NULL,
    title       TEXT,
    category    TEXT,
    description TEXT,      -- auto-generated or manually corrected caption
    embedding   BLOB       -- CLIP 512-dim float32 vector
)
"""

_CREATE_GOLDEN_TABLE = """
CREATE TABLE IF NOT EXISTS golden_test_set (
    id          TEXT PRIMARY KEY,
    asset_id    TEXT NOT NULL,
    caption     TEXT NOT NULL,    -- manually verified caption
    annotator   TEXT,
    created_at  TEXT DEFAULT (datetime('now'))
)
"""


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------
def init_db() -> None:
    """Create tables if they do not yet exist. Safe to call repeatedly."""
    with get_connection() as conn:
        conn.execute(_CREATE_ASSETS_TABLE)
        conn.execute(_CREATE_GOLDEN_TABLE)
        conn.commit()


# ---------------------------------------------------------------------------
# Connection helper
# ---------------------------------------------------------------------------
@contextmanager
def get_connection():
    """Context manager that yields a sqlite3.Connection and commits/rolls back."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# CRUD helpers
# ---------------------------------------------------------------------------

def insert_asset(asset_id: str, url: str, title: str, category: str,
                 description: str, embedding_bytes: bytes) -> None:
    """Insert or replace a single asset record."""
    with get_connection() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO assets
               (id, url, title, category, description, embedding)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (asset_id, url, title, category, description, embedding_bytes)
        )
        conn.commit()


def update_asset_description(asset_id: str, description: str) -> None:
    """Update the caption/description for an existing asset (annotation UI)."""
    with get_connection() as conn:
        conn.execute(
            "UPDATE assets SET description = ? WHERE id = ?",
            (description, asset_id)
        )
        conn.commit()


def get_all_assets() -> list[dict]:
    """Return all assets as a list of dicts (without embedding bytes)."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT id, url, title, category, description FROM assets"
        ).fetchall()
    return [dict(r) for r in rows]


def get_asset_by_id(asset_id: str) -> Optional[dict]:
    """Return a single asset dict including embedding bytes, or None."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT id, url, title, category, description, embedding FROM assets WHERE id = ?",
            (asset_id,)
        ).fetchone()
    return dict(row) if row else None


def get_all_assets_with_embeddings() -> list[dict]:
    """Return all assets including embedding blobs — used by search."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT id, url, title, category, description, embedding FROM assets"
        ).fetchall()
    return [dict(r) for r in rows]


def insert_golden_entry(entry_id: str, asset_id: str,
                        caption: str, annotator: str = "") -> None:
    """Save a manually verified golden test-set entry."""
    with get_connection() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO golden_test_set
               (id, asset_id, caption, annotator)
               VALUES (?, ?, ?, ?)""",
            (entry_id, asset_id, caption, annotator)
        )
        conn.commit()


def get_golden_test_set() -> list[dict]:
    """Return all golden test-set entries."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT id, asset_id, caption, annotator, created_at FROM golden_test_set"
        ).fetchall()
    return [dict(r) for r in rows]
