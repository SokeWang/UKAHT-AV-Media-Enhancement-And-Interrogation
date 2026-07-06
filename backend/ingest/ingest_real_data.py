"""
backend/ingest/ingest_real_data.py
Owner: Yisheng Zhang — Milestone 2 (batch ingestion of all UKAHT images)

Responsibilities:
  - Scan the UKAHT image directory for supported image files
  - Run BLIP to generate automatic captions in mini-batches
  - Run CLIP to compute image embeddings in mini-batches
  - Write each record (metadata + caption + embedding) to SQLite
  - Support --sample N flag for quick testing on a small subset
  - Support --batch-size flag to tune memory usage on different machines

Usage:
  python -m backend.ingest.ingest_real_data --dir "/path/to/UKAHT Data"
  python -m backend.ingest.ingest_real_data --dir "/path/to/UKAHT Data" --sample 20
  python -m backend.ingest.ingest_real_data --dir "/path/to/UKAHT Data" --batch-size 8
"""

import argparse
import os
import uuid
from pathlib import Path

from backend.db.database import init_db, insert_asset
from backend.models.blip_model import get_captions_batch
from backend.models.clip_model import get_image_embeddings_batch

# ---------------------------------------------------------------------------
# Supported image extensions
# ---------------------------------------------------------------------------
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


# ---------------------------------------------------------------------------
# File scanner
# ---------------------------------------------------------------------------

def scan_images(directory: str) -> list[Path]:
    """
    Recursively find all supported image files under a directory.

    Returns:
        Sorted list of Path objects.
    """
    root = Path(directory)
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {directory}")

    paths = sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    return paths


# ---------------------------------------------------------------------------
# Batch ingestion
# ---------------------------------------------------------------------------

def ingest_batch(directory: str, batch_size: int = 4, sample: int = 0) -> None:
    """
    Main ingestion routine.

    Args:
        directory:  Root directory of the UKAHT image archive.
        batch_size: Number of images processed together in one BLIP/CLIP call.
        sample:     If > 0, only process the first N images (for quick testing).
    """
    init_db()

    all_paths = scan_images(directory)
    if sample > 0:
        all_paths = all_paths[:sample]

    total = len(all_paths)
    print(f"Found {total} images under '{directory}'.")

    processed = 0
    for batch_start in range(0, total, batch_size):
        batch_paths = all_paths[batch_start: batch_start + batch_size]
        str_paths = [str(p) for p in batch_paths]

        # --- Captions -------------------------------------------------------
        try:
            captions = get_captions_batch(str_paths)
        except Exception as exc:
            print(f"  [WARN] BLIP failed on batch {batch_start}: {exc}")
            captions = [""] * len(batch_paths)

        # --- Embeddings -----------------------------------------------------
        try:
            embeddings = get_image_embeddings_batch(str_paths)
        except Exception as exc:
            print(f"  [WARN] CLIP failed on batch {batch_start}: {exc}")
            import numpy as np
            embeddings = [np.zeros(512, dtype="float32")] * len(batch_paths)

        # --- Write to DB ----------------------------------------------------
        for path, caption, emb in zip(batch_paths, captions, embeddings):
            asset_id = f"ukaht_{uuid.uuid4().hex[:10]}"
            title = path.stem.replace("_", " ").replace("-", " ").title()
            # Derive a coarse category from the parent directory name
            category = path.parent.name.replace("_", " ").title()

            insert_asset(
                asset_id=asset_id,
                url=str(path),          # absolute path — adjust if serving via HTTP
                title=title,
                category=category,
                description=caption,
                embedding_bytes=emb.astype("float32").tobytes(),
            )
            processed += 1

        print(f"  Processed {min(processed, total)}/{total} images …")

    print(f"Ingestion complete. {processed} assets written to database.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Batch ingest UKAHT images into SQLite using BLIP + CLIP."
    )
    parser.add_argument(
        "--dir",
        default=os.getenv(
            "UKAHT_DATA_DIR",
            "/Users/wangpeidong/UKAHT-Project/UK Antarctic Heritage Trust Data",
        ),
        help="Root directory of the UKAHT image archive",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="Number of images per BLIP/CLIP batch (default: 4)",
    )
    parser.add_argument(
        "--sample", type=int, default=0,
        help="Only process the first N images (0 = all)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    ingest_batch(
        directory=args.dir,
        batch_size=args.batch_size,
        sample=args.sample,
    )
