import sqlite3
import numpy as np
import os

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend", "db.sqlite")
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute("SELECT id, title, url, embedding FROM assets")
rows = cursor.fetchall()
for r in rows:
    emb = np.frombuffer(r[3], dtype=np.float32)
    print(f"ID: {r[0]}, Title: {r[1]}, URL: {r[2]}")
    print(f"  Embedding len: {len(emb)}, sum: {np.sum(emb)}, norm: {np.linalg.norm(emb)}")
    print(f"  First 5 values: {emb[:5]}")
conn.close()
