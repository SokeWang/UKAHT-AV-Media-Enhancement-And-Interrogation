"""
experiments/yisheng/update_db_with_gemma_captions.py

Batch Database Update Tool (Yisheng Zhang):
Updates PostgreSQL database table `assets` by replacing raw BLIP captions
with Gemma 4:e4b Normalized Multimodal Captions.

Executable via Docker:
  docker exec ukaht-algorithm python3 experiments/yisheng/update_db_with_gemma_captions.py
"""

import os
import sys
import json
import time
import psycopg2
from psycopg2.extras import RealDictCursor

# Ensure project root is in Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.yisheng.caption_annotator import GemmaVLMAnnotator


def run_batch_update():
    print("==================================================================")
    print("=== [Yisheng Data Science] Updating Database with Gemma Captions ===")
    print("==================================================================")

    # Check if pre-evaluated JSON results exist
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gemma_vs_blip_comparison.json")
    cached_captions = {}
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data.get("results", []):
                    cached_captions[item["asset_id"]] = item["gemma_e4b_caption"]
            print(f"Loaded {len(cached_captions)} pre-evaluated Gemma captions from '{json_path}'.")
        except Exception as exc:
            print(f"[WARN] Failed to load JSON cache: {exc}")

    annotator = GemmaVLMAnnotator()

    conn = psycopg2.connect(host="db", port=5432, dbname="ukaht", user="postgres", password="postgres")
    cur = conn.cursor(cursor_factory=RealDictCursor)

    cur.execute("SELECT id, title, category, description, base_code, subject_type, shooting_year FROM assets ORDER BY id ASC")
    rows = cur.fetchall()
    total = len(rows)

    print(f"Found {total} assets in PostgreSQL database. Starting batch description update...")

    updated_count = 0
    start_time = time.time()

    for idx, asset in enumerate(rows, 1):
        asset_id = asset["id"]
        
        # Use cached caption if available, otherwise generate via GemmaAnnotator
        if asset_id in cached_captions:
            gemma_caption = cached_captions[asset_id]
        else:
            gemma_caption = annotator.generate_caption(asset)

        # Update PostgreSQL assets table
        cur.execute(
            "UPDATE assets SET description = %s WHERE id = %s",
            (gemma_caption, asset_id)
        )
        conn.commit()
        updated_count += 1

        if idx % 10 == 0 or idx == total:
            print(f"[{idx}/{total}] Updated asset '{asset_id}' description in PostgreSQL database.")

    cur.close()
    conn.close()

    elapsed = time.time() - start_time
    print("\n==================================================================")
    print("=== POSTGRESQL DATABASE UPDATE COMPLETE ===")
    print(f"-> Total Assets Updated : {updated_count}/{total}")
    print(f"-> Execution Time       : {round(elapsed, 2)} seconds")
    print("-> Database Status      : All assets.description updated to Gemma 4:e4b Normalized Captions.")
    print("==================================================================")


if __name__ == "__main__":
    run_batch_update()
