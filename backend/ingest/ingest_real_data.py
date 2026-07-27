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
import json
import re
import requests
import numpy as np
from pathlib import Path
from typing import Optional

from backend.db.database import init_db, insert_asset

ALGO_API_BASE = os.getenv("ALGO_API_BASE", "http://localhost:8001")

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


def extract_metadata_via_llm(file_path: str) -> dict:
    """
    Use local LLM to extract metadata (base_code, subject_type, shooting_year, copyright)
    from the file path. Falls back to regex-based extraction if LLM fails or is unavailable.
    """
    path_obj = Path(file_path)
    
    # 1. Determine data source
    data_source = "original"
    for part in path_obj.parts:
        part_lower = part.lower()
        if "uk antarctic heritage trust data" in part_lower:
            data_source = "original"
            break
        elif "ukaht data" in part_lower:
            data_source = "new_addition"
            break

    # Get the key path part (parent directory name + file name)
    parent_name = path_obj.parent.name
    file_name = path_obj.name
    full_ident = f"Folder: {parent_name} / File: {file_name}"

    # Try calling LLM first
    try:
        model_name = os.getenv("UKAHT_LLM_MODEL", "")
        base_url = os.getenv("UKAHT_LLM_BASE_URL", "http://localhost:11434/v1")
        api_key = os.getenv("UKAHT_LLM_API_KEY", "ollama")

        system_message = (
            "You are an expert Antarctic archival metadata extractor. "
            "Given a directory name and file name of an image, extract the metadata: "
            "1. base_code: Single letter representing the Base (e.g., 'E' for Base E/Stonington, 'W' for Base W/Detaille, 'A' for Base A/Port Lockroy, or null if unknown).\n"
            "2. subject_type: A short capitalized phrase representing what the photo is about (e.g., 'Exterior', 'Main Hut', 'Artifact', 'SfM', 'Landscape', or null if unknown).\n"
            "3. shooting_year: The year or season the photo was taken (e.g. '1958', '2011-12', '2025', or null if unknown).\n"
            "4. copyright: The credit, copyright owner, or photographer name (e.g. 'Mike Cousins', 'Gordon MacDonald', 'Neil Marsden', 'JNH', or null if unknown).\n"
            "Respond ONLY with a raw JSON object containing these 4 keys: base_code, subject_type, shooting_year, copyright. Do not include markdown code block formatting or explanations."
        )

        human_message = f"Please extract metadata for:\n{full_ident}"
        
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": human_message}
            ],
            "temperature": 0
        }
        headers = {"Authorization": f"Bearer {api_key}"}
        resp = requests.post(f"{base_url}/chat/completions", json=payload, headers=headers, timeout=120.0)
        resp.raise_for_status()
        raw_content = resp.json()["choices"][0]["message"]["content"].strip()
        
        # Clean up any potential markdown formatting
        if raw_content.startswith("```"):
            lines = raw_content.splitlines()
            if lines[0].startswith("```json"):
                raw_content = "\n".join(lines[1:-1]).strip()
            elif lines[0].startswith("```"):
                raw_content = "\n".join(lines[1:-1]).strip()

        parsed = json.loads(raw_content)
        return {
            "base_code": parsed.get("base_code"),
            "subject_type": parsed.get("subject_type"),
            "shooting_year": parsed.get("shooting_year"),
            "copyright": parsed.get("copyright"),
            "data_source": data_source
        }
    except Exception as exc:
        print(f"  [INFO] LLM metadata extraction failed or offline, falling back to regex: {exc}")
        return fallback_extract_metadata(file_path, data_source)


def fallback_extract_metadata(file_path: str, data_source: str) -> dict:
    """
    Regex-based fallback parser in case the LLM is offline or fails.
    """
    path_obj = Path(file_path)
    target_dir_name = ""
    for parent in [path_obj.stem] + [p.name for p in path_obj.parents]:
        if "_" in parent or any(x in parent.lower() for x in ["copyright", "copyrright", "cr_"]):
            target_dir_name = parent
            break
            
    if not target_dir_name:
        return {
            "base_code": None, "subject_type": None, "shooting_year": None, 
            "copyright": None, "data_source": data_source
        }

    # Extract Copyright
    copyright_val = None
    copyright_pattern = r'(?i)(?:cr|copyright|copyrright)_(.*)'
    copyright_match = re.search(copyright_pattern, target_dir_name)
    if copyright_match:
        copyright_val = copyright_match.group(1).replace("_", " ").title()
        clean_dir_name = re.sub(copyright_pattern, '', target_dir_name).rstrip('_')
    else:
        clean_dir_name = target_dir_name

    # Extract year, base, subject
    tokens = [t for t in clean_dir_name.split("_") if t]
    base_code = None
    shooting_year = None
    subject_parts = []

    year_regex = re.compile(r'^\d{4}(?:[-_]\d{2,4})?$')
    base_regex = re.compile(r'^[a-zA-Z]$')

    for token in tokens:
        token_strip = token.strip()
        if year_regex.match(token_strip):
            shooting_year = token_strip.replace("-", "_")
        elif base_regex.match(token_strip):
            base_code = token_strip.upper()
        else:
            split_camel = re.sub(r'([a-z])([A-Z])', r'\1 \2', token_strip)
            subject_parts.append(split_camel.title())

    subject_type = " ".join(subject_parts) if subject_parts else None

    # Handle old dataset specific cases if needed
    path_lower = file_path.lower()
    if not base_code:
        if "base e" in path_lower or "hist_e" in path_lower:
            base_code = "E"
        elif "base w" in path_lower:
            base_code = "W"
        elif "various_a" in path_lower or "artefacts_a" in path_lower or "port lockroy" in path_lower:
            base_code = "A"

    if not copyright_val:
        if "neilmarsden" in path_lower or "marsden, neil" in path_lower:
            copyright_val = "Neil Marsden"
        elif "copyright_jnh" in path_lower:
            copyright_val = "JNH"
        elif "copyrright_l_ling" in path_lower:
            copyright_val = "L. Ling"

    return {
        "base_code": base_code,
        "subject_type": subject_type,
        "shooting_year": shooting_year,
        "copyright": copyright_val,
        "data_source": data_source
    }


# ---------------------------------------------------------------------------
# S3 scanner helper
# ---------------------------------------------------------------------------

def scan_images_s3(bucket_name: str, prefix: str = "", region: Optional[str] = None) -> list[str]:
    """
    Recursively list all supported image files under a prefix in an S3 bucket.
    """
    import boto3
    s3 = boto3.client("s3", region_name=region) if region else boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    
    keys = []
    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if "s3_temp" in key or "temp/" in key or "tmp/" in key:
                continue
            ext = os.path.splitext(key)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                keys.append(key)
    return sorted(keys)


# ---------------------------------------------------------------------------
# Batch ingestion
# ---------------------------------------------------------------------------

def ingest_batch(
    directory: Optional[str] = None,
    s3_bucket: Optional[str] = None,
    s3_prefix: str = "",
    s3_region: Optional[str] = None,
    batch_size: int = 4,
    sample: int = 0
) -> None:
    """
    Main ingestion routine. Can ingest from local filesystem or AWS S3.
    """
    import boto3
    import shutil
    init_db()

    is_s3 = bool(s3_bucket)
    if is_s3:
        print(f"Scanning S3 bucket '{s3_bucket}' with prefix '{s3_prefix}'...")
        all_keys = scan_images_s3(s3_bucket, s3_prefix, s3_region)
        if sample > 0:
            all_keys = all_keys[:sample]
        total = len(all_keys)
        print(f"Found {total} images in S3 bucket.")
        
        # Prepare local temp directory for batch downloads (shared with algorithm container)
        temp_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
            "static", "uploads", "s3_temp"
        )
        os.makedirs(temp_dir, exist_ok=True)
        s3_client = boto3.client("s3", region_name=s3_region) if s3_region else boto3.client("s3")
    else:
        if not directory:
            directory = "/Users/wangpeidong/UKAHT-Project/UK Antarctic Heritage Trust Data"
        print(f"Scanning local directory '{directory}'...")
        all_paths = scan_images(directory)
        if sample > 0:
            all_paths = all_paths[:sample]
        total = len(all_paths)
        print(f"Found {total} local images.")

    processed = 0
    for batch_start in range(0, total, batch_size):
        local_paths = []
        item_keys = []  # original path strings or S3 keys
        s3_urls = []

        if is_s3:
            batch_keys = all_keys[batch_start: batch_start + batch_size]
            for key in batch_keys:
                ext = os.path.splitext(key)[1].lower()
                filename = f"{uuid.uuid4().hex[:12]}{ext}"
                local_path = os.path.join(temp_dir, filename)
                try:
                    s3_client.download_file(s3_bucket, key, local_path)
                    local_paths.append(local_path)
                    item_keys.append(key)
                    
                    # Formulate S3 HTTP URL
                    region_str = f".{s3_region}" if s3_region else ""
                    s3_url = f"https://{s3_bucket}.s3{region_str}.amazonaws.com/{key}"
                    s3_urls.append(s3_url)
                except Exception as exc:
                    print(f"  [ERROR] Failed to download {key} from S3: {exc}")
        else:
            batch_paths = all_paths[batch_start: batch_start + batch_size]
            for p in batch_paths:
                local_paths.append(str(p))
                item_keys.append(str(p))
                s3_urls.append(str(p))

        if not local_paths:
            continue

        # --- Captions -------------------------------------------------------
        try:
            resp = requests.post(f"{ALGO_API_BASE}/api/algo/caption/batch", json={"paths": local_paths}, timeout=60)
            resp.raise_for_status()
            captions = resp.json()["data"]["captions"]
        except Exception as exc:
            print(f"  [WARN] BLIP failed on batch {batch_start}: {exc}")
            captions = [""] * len(local_paths)

        # --- Embeddings -----------------------------------------------------
        try:
            resp = requests.post(f"{ALGO_API_BASE}/api/algo/embed/image/batch", json={"paths": local_paths}, timeout=60)
            resp.raise_for_status()
            embeddings_list = resp.json()["data"]["embeddings"]
            embeddings = [np.array(emb, dtype=np.float32) for emb in embeddings_list]
        except Exception as exc:
            print(f"  [WARN] CLIP failed on batch {batch_start}: {exc}")
            embeddings = [np.zeros(512, dtype="float32")] * len(local_paths)

        # --- Write to DB & Clean up -----------------------------------------
        for idx, (path_or_key, caption, emb, url) in enumerate(zip(item_keys, captions, embeddings, s3_urls)):
            # Check for duplicate image in the database
            from backend.db.database import check_duplicate_image
            emb_bytes = emb.astype("float32").tobytes()
            duplicate_id = check_duplicate_image(emb_bytes)
            if duplicate_id:
                print(f"  [INFO] Visual duplicate detected for {path_or_key} (matches asset: {duplicate_id}). Skipping database insertion.")
                processed += 1
                # Clean up local temp files immediately to free space
                if is_s3:
                    try:
                        os.remove(local_paths[idx])
                    except Exception:
                        pass
                continue

            asset_id = f"ukaht_{uuid.uuid4().hex[:10]}"
            
            # Formulate friendly title from the stem of original filename
            title_stem = Path(path_or_key).stem
            title = title_stem.replace("_", " ").replace("-", " ").title()
            
            # --- Extract Metadata (pass original path/key to preserve structure) ---
            meta = extract_metadata_via_llm(path_or_key)
            category = meta["subject_type"] or Path(path_or_key).parent.name.replace("_", " ").title()

            insert_asset(
                asset_id=asset_id,
                url=url,                # S3 URL if S3, otherwise local file path
                title=title,
                category=category,
                description=caption,
                embedding_bytes=emb_bytes,
                base_code=meta["base_code"],
                subject_type=meta["subject_type"],
                shooting_year=meta["shooting_year"],
                copyright=meta["copyright"],
                data_source=meta["data_source"]
            )
            processed += 1

            # Clean up local temp files immediately to free space
            if is_s3:
                try:
                    os.remove(local_paths[idx])
                except Exception:
                    pass

        print(f"  Processed {min(processed, total)}/{total} images …")

    # Clean up temp folder completely
    if is_s3:
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass

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
        default=None,
        help="Root directory of the local UKAHT image archive",
    )
    parser.add_argument(
        "--s3-bucket",
        default=os.getenv("UKAHT_S3_BUCKET"),
        help="AWS S3 Bucket name containing images",
    )
    parser.add_argument(
        "--s3-prefix",
        default=os.getenv("UKAHT_S3_PREFIX", ""),
        help="Prefix filter for S3 bucket objects",
    )
    parser.add_argument(
        "--s3-region",
        default=os.getenv("UKAHT_S3_REGION"),
        help="AWS S3 Bucket region name",
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
        s3_bucket=args.s3_bucket,
        s3_prefix=args.s3_prefix,
        s3_region=args.s3_region,
        batch_size=args.batch_size,
        sample=args.sample,
    )
