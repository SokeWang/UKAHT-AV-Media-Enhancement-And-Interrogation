"""
experiments/yisheng/sync_framework.py

Object-Oriented S3 Database Synchronization & Ingestion Framework (Yisheng Zhang)
Provides S3DatabaseSynchronizer class for scanning S3, batch processing, multimodal feature extraction,
annotator invocation, and PostgreSQL asset persistence.
"""

import os
import sys
import uuid
import time
import requests
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.db.database import get_connection, insert_asset
from backend.ingest.ingest_real_data import extract_metadata_via_llm, scan_images_s3
from experiments.yisheng.caption_annotator import BaseAnnotator, build_annotator, get_annotator_from_config


class S3DatabaseSynchronizer:
    """
    Object-Oriented Framework Manager for S3 and Database Synchronization.
    """
    def __init__(
        self,
        s3_bucket: Optional[str] = None,
        s3_prefix: str = "",
        s3_region: Optional[str] = None,
        local_dir: Optional[str] = None,
        annotator: Optional[BaseAnnotator] = None,
        algo_base_url: Optional[str] = None
    ):
        self.s3_bucket = s3_bucket or os.getenv("UKAHT_S3_BUCKET")
        self.s3_prefix = s3_prefix or os.getenv("UKAHT_S3_PREFIX", "")
        self.s3_region = s3_region or os.getenv("UKAHT_S3_REGION", "us-east-1")
        self.local_dir = local_dir or os.getenv("UKAHT_DATA_DIR")
        self.annotator = annotator or get_annotator_from_config()
        self.algo_base_url = algo_base_url or os.getenv("ALGO_API_BASE", "http://localhost:8001")

        self.is_s3 = bool(self.s3_bucket)
        self.stats = {
            "processed_count": 0,
            "duplicate_count": 0,
            "failed_count": 0,
            "total_images_scanned": 0
        }

    def scan_sources(self) -> List[str]:
        """Scan S3 bucket or local directory for supported image assets."""
        if self.is_s3:
            print(f"[S3SyncFramework] Scanning S3 Bucket '{self.s3_bucket}' (Prefix: '{self.s3_prefix}')...")
            keys = scan_images_s3(self.s3_bucket, self.s3_prefix, self.s3_region)
            self.stats["total_images_scanned"] = len(keys)
            return keys
        else:
            print(f"[S3SyncFramework] Scanning Local Directory '{self.local_dir}'...")
            if not self.local_dir or not os.path.exists(self.local_dir):
                print(f"[WARN] Local directory '{self.local_dir}' not found.")
                return []
            exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
            p_list = [str(p) for p in Path(self.local_dir).rglob("*") if p.suffix.lower() in exts]
            self.stats["total_images_scanned"] = len(p_list)
            return p_list

    def extract_visual_embeddings(self, local_paths: List[str]) -> List[np.ndarray]:
        """Extract CLIP visual embeddings via algorithm service."""
        try:
            resp = requests.post(f"{self.algo_base_url}/api/algo/embed/image/batch", json={"paths": local_paths}, timeout=60)
            resp.raise_for_status()
            embeddings_list = resp.json()["data"]["embeddings"]
            return [np.array(emb, dtype=np.float32) for emb in embeddings_list]
        except Exception as exc:
            print(f"[S3SyncFramework WARN] CLIP Embedding extraction failed: {exc}")
            return [np.zeros(512, dtype="float32") for _ in local_paths]

    def sync(self, batch_size: int = 4, sample_limit: int = 0) -> Dict[str, Any]:
        """
        Execute full batch synchronization and database ingestion pipeline.
        """
        print("==================================================================")
        print("=== [S3SyncFramework] Executing Framework Sync Pipeline ===")
        print("==================================================================")

        all_keys = self.scan_sources()
        if sample_limit > 0:
            all_keys = all_keys[:sample_limit]

        total = len(all_keys)
        print(f"[S3SyncFramework] Beginning ingestion for {total} image assets (Batch Size: {batch_size})...")

        start_time = time.time()

        for batch_start in range(0, total, batch_size):
            batch_keys = all_keys[batch_start: batch_start + batch_size]
            local_paths = []
            s3_urls = []
            item_keys = []

            # Resolve S3 download or local paths
            if self.is_s3:
                import boto3
                temp_dir = os.path.join(PROJECT_ROOT, "backend", "static", "uploads", "s3_temp")
                os.makedirs(temp_dir, exist_ok=True)
                s3_client = boto3.client("s3", region_name=self.s3_region)

                for key in batch_keys:
                    safe_filename = key.replace("/", "_")
                    local_path = os.path.join(temp_dir, safe_filename)
                    try:
                        s3_client.download_file(self.s3_bucket, key, local_path)
                        local_paths.append(local_path)
                        item_keys.append(key)
                        region_str = f".{self.s3_region}" if self.s3_region else ""
                        s3_urls.append(f"https://{self.s3_bucket}.s3{region_str}.amazonaws.com/{key}")
                    except Exception as exc:
                        print(f"  [ERROR] Failed to download {key} from S3: {exc}")
                        self.stats["failed_count"] += 1
            else:
                for p in batch_keys:
                    local_paths.append(str(p))
                    item_keys.append(str(p))
                    s3_urls.append(str(p))

            if not local_paths:
                continue

            # Extract CLIP visual embeddings
            embeddings = self.extract_visual_embeddings(local_paths)

            # Generate captions using configured Annotator class
            meta_assets = []
            for path_or_key, local_p, url in zip(item_keys, local_paths, s3_urls):
                meta = extract_metadata_via_llm(path_or_key)
                title_stem = Path(path_or_key).stem
                meta_assets.append({
                    "title": title_stem.replace("_", " ").replace("-", " ").title(),
                    "category": meta["subject_type"] or "Antarctic Heritage",
                    "base_code": meta["base_code"],
                    "subject_type": meta["subject_type"],
                    "shooting_year": meta["shooting_year"],
                    "copyright": meta["copyright"],
                    "data_source": meta["data_source"],
                    "local_path": local_p,
                    "url": url,
                    "key": path_or_key
                })

            captions = self.annotator.batch_generate_captions(meta_assets)

            # Insert into PostgreSQL database
            for meta_item, caption, emb in zip(meta_assets, captions, embeddings):
                asset_id = f"ukaht_{uuid.uuid4().hex[:10]}"
                emb_bytes = emb.astype("float32").tobytes()

                try:
                    insert_asset(
                        asset_id=asset_id,
                        url=meta_item["url"],
                        title=meta_item["title"],
                        category=meta_item["category"],
                        description=caption,
                        embedding_bytes=emb_bytes,
                        base_code=meta_item["base_code"],
                        subject_type=meta_item["subject_type"],
                        shooting_year=meta_item["shooting_year"],
                        copyright=meta_item["copyright"],
                        data_source=meta_item["data_source"]
                    )
                    self.stats["processed_count"] += 1
                except Exception as exc:
                    print(f"  [ERROR] Failed DB insertion for {meta_item['key']}: {exc}")
                    self.stats["failed_count"] += 1

                # Clean up local S3 temp files
                if self.is_s3 and os.path.exists(meta_item["local_path"]):
                    try:
                        os.remove(meta_item["local_path"])
                    except Exception:
                        pass

            print(f"[S3SyncFramework] Progress: [{min(batch_start + batch_size, total)}/{total}] synchronized to DB.")

        elapsed = time.time() - start_time
        self.stats["elapsed_seconds"] = round(elapsed, 2)

        print("\n==================================================================")
        print("=== [S3SyncFramework] SYNC PIPELINE COMPLETE ===")
        print(f"-> Total Ingested : {self.stats['processed_count']}")
        print(f"-> Total Failed   : {self.stats['failed_count']}")
        print(f"-> Total Elapsed  : {self.stats['elapsed_seconds']} s")
        print("==================================================================")

        return self.stats
