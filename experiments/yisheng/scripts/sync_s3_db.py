"""
experiments/yisheng/scripts/sync_s3_db.py

Standalone Experiment & Pipeline Runner: S3 Database Sync Framework
Calls S3DatabaseSynchronizer and Annotator classes defined under experiments/yisheng/

Usage via Docker:
  docker exec ukaht-algorithm python3 experiments/yisheng/scripts/sync_s3_db.py --sample 20
  docker exec ukaht-backend python3 experiments/yisheng/scripts/sync_s3_db.py --s3-bucket my-bucket --use-gemma
"""

import sys
import os
import argparse

# Ensure project root is in Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.yisheng.sync_framework import S3DatabaseSynchronizer
from experiments.yisheng.caption_annotator import build_annotator


def main():
    parser = argparse.ArgumentParser(
        description="Run S3 Database Synchronization Framework (Yisheng Zhang)"
    )
    parser.add_argument(
        "--s3-bucket",
        default=os.getenv("UKAHT_S3_BUCKET"),
        help="AWS S3 Bucket name (default: env UKAHT_S3_BUCKET)",
    )
    parser.add_argument(
        "--s3-prefix",
        default=os.getenv("UKAHT_S3_PREFIX", ""),
        help="S3 key prefix filter (default: env UKAHT_S3_PREFIX)",
    )
    parser.add_argument(
        "--s3-region",
        default=os.getenv("UKAHT_S3_REGION", "us-east-1"),
        help="AWS S3 region (default: env UKAHT_S3_REGION or us-east-1)",
    )
    parser.add_argument(
        "--dir",
        default=None,
        help="Local directory override",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size (default: 4)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Limit number of sample images to sync (0 = all)",
    )
    parser.add_argument(
        "--annotator",
        choices=["gemma", "blip"],
        default="gemma",
        help="Annotator model framework to use (default: gemma)",
    )

    args = parser.parse_args()

    # Instantiate object-oriented Annotator & Synchronizer framework
    annotator_instance = build_annotator(mode=args.annotator)
    synchronizer = S3DatabaseSynchronizer(
        s3_bucket=args.s3_bucket,
        s3_prefix=args.s3_prefix,
        s3_region=args.s3_region,
        local_dir=args.dir,
        annotator=annotator_instance
    )

    # Execute framework sync
    synchronizer.sync(
        batch_size=args.batch_size,
        sample_limit=args.sample
    )


if __name__ == "__main__":
    main()
