"""
experiments/peidong/scripts/exp07_gemma_crossmodal_adapter.py

Standalone Experiment Runner: Phase 8 — Gemma-Guided Cross-Modal Contrastive Alignment Innovation.
Leverages Gemma 4:e4b Normalized Captions + Base Code + Category metadata as cross-modal positive supervision.

Usage:
  docker exec ukaht-algorithm python3 experiments/peidong/scripts/exp07_gemma_crossmodal_adapter.py --epochs 30 --mode dual_swiglu
"""

import os
import sys
import argparse

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.peidong.train_adapter import train


def main():
    parser = argparse.ArgumentParser(
        description="Run Experiment 07: Gemma-Guided Cross-Modal Contrastive Polar Adapter Alignment"
    )
    parser.add_argument("--mode", type=str, default="dual_swiglu", choices=["dual_swiglu", "pcse", "mlp", "ted", "lora"], help="Adapter mode")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    args = parser.parse_args()

    print("==========================================================================================")
    print("=== PEIDONG EXPERIMENT 07: GEMMA-GUIDED CROSS-MODAL CONTRASTIVE POLAR ADAPTER ===")
    print("==========================================================================================")

    output_path = "/app/backend/static/models/adapter_gemma_crossmodal.pth"

    train(
        mode=args.mode,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        loss_type="infonce",
        use_gemma_contrastive=True,
        output_path=output_path
    )


if __name__ == "__main__":
    main()
