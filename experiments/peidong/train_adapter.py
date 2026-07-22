"""
experiments/peidong/train_adapter.py
Owner: Peidong Wang — Milestone 3 (Polar Domain Adapter Fine-Tuning Script)

Features:
- Configurable Fine-Tuning Mode: --mode [mlp | lora | qlora]
- Automatic Database Connection: Connects to PostgreSQL or SQLite db
- Triplet Loss Training: Constructs (Anchor, Positive, Negative) embedding pairs
- Saves trained weights with auto-detection metadata to static/models/adapter.pth

Usage:
  Python on host / local:
    python3 experiments/peidong/train_adapter.py --mode lora --epochs 20

  Docker on server / dev machine:
    docker exec -it ukaht-algorithm python3 /app/experiments/peidong/train_adapter.py --mode mlp
"""

import os
import sys
import argparse
import random
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    print("[ERROR] PyTorch is required to run training. Please run inside the algorithm container or install torch.")
    sys.exit(1)

# Import our modular adapter
try:
    from experiments.peidong.adapter import _build_model, save_adapter
except ImportError:
    from algorithm.models.adapter import _build_model, save_adapter


def load_dataset_embeddings():
    """Load all asset embeddings and categories from database (PostgreSQL or SQLite)."""
    assets = []
    
    # Try PostgreSQL across common docker/local hosts and db names
    hosts_to_try = []
    if os.getenv("POSTGRES_HOST"):
        hosts_to_try.append(os.getenv("POSTGRES_HOST"))
    hosts_to_try.extend(["db", "ukaht-db", "localhost", "127.0.0.1"])
    
    db_names_to_try = [os.getenv("POSTGRES_DB", "ukaht"), "ukaht", "ukaht_db"]
    pg_user = os.getenv("POSTGRES_USER", "postgres")
    pg_pass = os.getenv("POSTGRES_PASSWORD", "postgres")
    pg_port = int(os.getenv("POSTGRES_PORT", 5432))
    
    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor
        
        for host in hosts_to_try:
            for db_name in db_names_to_try:
                try:
                    conn = psycopg2.connect(
                        host=host, 
                        database=db_name, 
                        user=pg_user, 
                        password=pg_pass, 
                        port=pg_port, 
                        connect_timeout=3
                    )
                    cursor = conn.cursor(cursor_factory=RealDictCursor)
                    cursor.execute("SELECT id, title, category, base_code, embedding FROM assets WHERE embedding IS NOT NULL")
                    rows = cursor.fetchall()
                    conn.close()
                    
                    if rows:
                        for row in rows:
                            emb_bytes = row["embedding"]
                            if emb_bytes is not None:
                                emb = np.frombuffer(emb_bytes, dtype=np.float32)
                                assets.append({
                                    "id": row["id"],
                                    "category": row["category"] or "default",
                                    "base_code": row["base_code"] or "default",
                                    "embedding": emb
                                })
                        print(f"[DATA] Successfully loaded {len(assets)} embeddings from PostgreSQL database at '{host}:{pg_port}/{db_name}'.")
                        return assets
                except Exception:
                    continue
    except ImportError:
        print("[WARN] psycopg2 module not available.")

    # Fallback to SQLite if PostgreSQL did not return embeddings
    sqlite_paths = [
        os.path.join(PROJECT_ROOT, "backend", "db.sqlite"),
        os.path.join(PROJECT_ROOT, "backend", "db", "db.sqlite"),
        "/app/backend/db.sqlite",
        "/app/backend/db/db.sqlite"
    ]
    for db_path in sqlite_paths:
        if os.path.exists(db_path):
            try:
                import sqlite3
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT id, title, category, base_code, embedding FROM assets WHERE embedding IS NOT NULL")
                rows = cursor.fetchall()
                conn.close()
                if rows:
                    for row in rows:
                        emb_bytes = row[4]
                        if emb_bytes is not None:
                            emb = np.frombuffer(emb_bytes, dtype=np.float32)
                            assets.append({
                                "id": row[0],
                                "category": row[2] or "default",
                                "base_code": row[3] or "default",
                                "embedding": emb
                            })
                    print(f"[DATA] Loaded {len(assets)} embeddings from SQLite database ('{db_path}').")
                    return assets
            except Exception as sqlite_err:
                print(f"[WARN] Failed reading SQLite at {db_path}: {sqlite_err}")

    return assets


class TripletDataset(Dataset):
    """Constructs (Anchor, Positive, Negative) embedding triplets for metric learning."""
    def __init__(self, assets, num_samples=1000):
        self.assets = assets
        self.num_samples = num_samples
        
        # Group by category / base_code
        self.groups = {}
        for idx, a in enumerate(assets):
            cat = a["category"]
            if cat not in self.groups:
                self.groups[cat] = []
            self.groups[cat].append(idx)
            
        self.categories = list(self.groups.keys())

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Pick anchor category
        if len(self.categories) > 1:
            cat_a = random.choice([c for c in self.categories if len(self.groups[c]) >= 1])
        else:
            cat_a = self.categories[0]
            
        # Pick anchor
        anchor_idx = random.choice(self.groups[cat_a])
        anchor_emb = self.assets[anchor_idx]["embedding"]
        
        # Pick positive (same category or corrupted version of anchor)
        if len(self.groups[cat_a]) > 1:
            pos_idx = random.choice([i for i in self.groups[cat_a] if i != anchor_idx])
            pos_emb = self.assets[pos_idx]["embedding"]
        else:
            # Add slight Gaussian noise to create synthetic positive
            noise = np.random.normal(0, 0.02, anchor_emb.shape).astype(np.float32)
            pos_emb = anchor_emb + noise
            pos_emb = pos_emb / np.linalg.norm(pos_emb)

        # Pick negative (different category or distant vector)
        diff_cats = [c for c in self.categories if c != cat_a]
        if diff_cats:
            cat_n = random.choice(diff_cats)
            neg_idx = random.choice(self.groups[cat_n])
            neg_emb = self.assets[neg_idx]["embedding"]
        else:
            # Pick a random vector with negative direction
            neg_emb = -anchor_emb + np.random.normal(0, 0.1, anchor_emb.shape).astype(np.float32)
            neg_emb = neg_emb / np.linalg.norm(neg_emb)

        return (
            torch.tensor(anchor_emb, dtype=torch.float32),
            torch.tensor(pos_emb, dtype=torch.float32),
            torch.tensor(neg_emb, dtype=torch.float32)
        )


def train(mode="mlp", epochs=15, batch_size=16, lr=1e-4, lora_r=16, lora_alpha=32, output_path=None):
    print("=" * 60)
    print(f"  PEIDONG WANG - ADAPTER FINE-TUNING ({mode.upper()})")
    print("=" * 60)

    assets = load_dataset_embeddings()
    if not assets:
        print("[ERROR] No embeddings found in database. Cannot perform fine-tuning.")
        return

    # Dataset & DataLoader
    dataset = TripletDataset(assets, num_samples=max(500, len(assets) * 50))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Build model
    model = _build_model(mode=mode, lora_r=lora_r, lora_alpha=lora_alpha)
    model.train()

    criterion = nn.TripletMarginLoss(margin=0.2, p=2)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    print(f"[MODEL] Fine-tuning Mode: {mode.upper()}")
    print(f"[TRAIN] Total Train Samples per Epoch: {len(dataset)}, Epochs: {epochs}, Batch Size: {batch_size}, LR: {lr}")

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        batches = 0
        for anchor, pos, neg in loader:
            optimizer.zero_grad()
            
            out_a = model(anchor)
            out_p = model(pos)
            out_n = model(neg)
            
            loss = criterion(out_a, out_p, out_n)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batches += 1
            
        avg_loss = total_loss / max(1, batches)
        if epoch % max(1, epochs // 5) == 0 or epoch == epochs:
            print(f"  Epoch [{epoch:2d}/{epochs:2d}] -> Triplet Loss: {avg_loss:.6f}")

    # Determine default save path
    if not output_path:
        possible_paths = [
            os.path.join(PROJECT_ROOT, "backend", "static", "models", "adapter.pth"),
            "/app/backend/static/models/adapter.pth",
        ]
        output_path = possible_paths[0]

    save_adapter(model, output_path)
    print("=" * 60)
    print(f"[SUCCESS] Adapter weights ({mode.upper()}) saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Peidong's Polar Domain Adapter")
    parser.add_argument("--mode", type=str, default="mlp", choices=["mlp", "lora", "qlora"], help="Adapter architecture mode")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha scaling factor")
    parser.add_argument("--output-path", type=str, default="", help="Target weights file path (.pth)")
    args = parser.parse_args()

    out_path = args.output_path if args.output_path else None

    train(
        mode=args.mode,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        output_path=out_path
    )
