"""
experiments/peidong/scripts/exp06_ted_adapter_innovation.py
Standalone Experiment Runner: Phase 8 — TED-Adapter Disentangled Subspace Innovation.

Usage:
  python experiments/peidong/scripts/exp06_ted_adapter_innovation.py --epochs 60
"""
import sys, os, argparse
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from algorithm.train_adapter import train


def main():
    parser = argparse.ArgumentParser(description="Run Experiment 06: TED-Adapter Innovation")
    parser.add_argument("--epochs", type=int, default=60, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()

    print("=" * 80)
    print("  RUNNING EXPERIMENT 06: TED-ADAPTER DISENTANGLED SUBSPACE INNOVATION")
    print("=" * 80)

    train(
        mode="ted",
        epochs=args.epochs,
        lr=args.lr,
        loss_type="infonce",
        use_learnable_tau=True,
        use_pcse_loss=True,
        output_path="/app/backend/static/models/adapter_ted.pth"
    )


if __name__ == "__main__":
    main()
