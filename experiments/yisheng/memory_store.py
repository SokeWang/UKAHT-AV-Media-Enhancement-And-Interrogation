"""
experiments/yisheng/memory_store.py
Owner: Yisheng Zhang — Agent Memory Experiment

Responsibilities:
  - Provide short-term session memory for multi-turn archive exploration.
  - Provide lightweight persistent long-term memory using SQLite.
  - Retrieve relevant memories without introducing production dependencies.
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable


# ---------------------------------------------------------------------------
# Small text-similarity helper
# ---------------------------------------------------------------------------

def _tokens(text: str) -> set[str]:
    """Simple dependency-free tokenizer used for lightweight memory retrieval."""
    return {
        token.strip(".,!?;:()[]{}\"'").lower()
        for token in text.split()
        if token.strip(".,!?;:()[]{}\"'")
    }


def lexical_similarity(a: str, b: str) -> float:
    """Jaccard similarity in [0, 1]."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


# ---------------------------------------------------------------------------
# Short-term memory
# ---------------------------------------------------------------------------

class ShortTermMemory:
    """
    In-memory rolling conversation history.

    Stores the most recent N user/assistant turns. This is deliberately small
    so that the experiment can compare no-memory vs. short-term-memory
    conditions without requiring a full agent framework.
    """

    def __init__(self, max_messages: int = 8):
        self.max_messages = max_messages
        self._messages: deque[dict[str, Any]] = deque(maxlen=max_messages)

    def add(self, role: str, content: str, metadata: dict[str, Any] | None = None) -> None:
        self._messages.append(
            {
                "role": role,
                "content": content,
                "metadata": metadata or {},
                "timestamp": time.time(),
            }
        )

    def clear(self) -> None:
        self._messages.clear()

    def get_messages(self) -> list[dict[str, Any]]:
        return list(self._messages)

    def as_prompt_text(self) -> str:
        """Render recent messages as compact context for an LLM."""
        lines = []
        for item in self._messages:
            role = item["role"].upper()
            lines.append(f"{role}: {item['content']}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Long-term persistent memory
# ---------------------------------------------------------------------------

class LongTermMemory:
    """
    Persistent memory backed by SQLite.

    Intended for experiments only. It can store:
      - previously explored bases / sites
      - selected archive asset IDs
      - verified entity names
      - corrected captions
      - useful cross-session facts
    """

    def __init__(self, db_path: str | Path = "experiments/yisheng/memory_experiment.sqlite"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    namespace TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at REAL NOT NULL
                )
                """
            )

    def add(
        self,
        content: str,
        namespace: str = "default",
        metadata: dict[str, Any] | None = None,
    ) -> int:
        with self._connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO memories(namespace, content, metadata_json, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    namespace,
                    content,
                    json.dumps(metadata or {}, ensure_ascii=False),
                    time.time(),
                ),
            )
            return int(cursor.lastrowid)

    def list_all(self, namespace: str | None = None) -> list[dict[str, Any]]:
        with self._connection() as conn:
            if namespace is None:
                rows = conn.execute(
                    "SELECT * FROM memories ORDER BY created_at DESC"
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM memories
                    WHERE namespace = ?
                    ORDER BY created_at DESC
                    """,
                    (namespace,),
                ).fetchall()

        result = []
        for row in rows:
            item = dict(row)
            item["metadata"] = json.loads(item.pop("metadata_json"))
            result.append(item)
        return result

    def search(
        self,
        query: str,
        namespace: str | None = None,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[dict[str, Any]]:
        """
        Retrieve memories by simple lexical similarity.

        This keeps the experiment reproducible and dependency-free. It can later
        be replaced by CLIP/text embeddings or a vector database.
        """
        candidates = self.list_all(namespace=namespace)

        scored = []
        for item in candidates:
            score = lexical_similarity(query, item["content"])
            if score >= min_score:
                item = dict(item)
                item["score"] = score
                scored.append(item)

        scored.sort(key=lambda x: (x["score"], x["created_at"]), reverse=True)
        return scored[:top_k]

    def clear(self, namespace: str | None = None) -> None:
        with self._connection() as conn:
            if namespace is None:
                conn.execute("DELETE FROM memories")
            else:
                conn.execute("DELETE FROM memories WHERE namespace = ?", (namespace,))
