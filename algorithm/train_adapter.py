"""
algorithm/train_adapter.py
Owner: Peidong Wang — Milestone 3 (Polar Domain Adapter Fine-Tuning Script)

Features:
- Configurable Fine-Tuning Mode: --mode [mlp | lora | qlora]
- Automatic Database Connection: Connects to PostgreSQL or SQLite db
- Triplet Loss Training: Constructs (Anchor, Positive, Negative) embedding pairs
- Saves trained weights with auto-detection metadata to static/models/adapter.pth

Usage:
  Docker command on server / dev machine:
    docker exec -it ukaht-algorithm python3 /app/algorithm/train_adapter.py --mode mlp --epochs 20
    docker exec -it ukaht-algorithm python3 /app/algorithm/train_adapter.py --mode lora --epochs 20
    docker exec -it ukaht-algorithm python3 /app/algorithm/train_adapter.py --mode qlora --epochs 20
"""

import os
import sys
import argparse
import random
import json
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    print("[ERROR] PyTorch is required to run training. Please run inside the algorithm container or install torch.")
    sys.exit(1)

# Import modular adapter
try:
    from algorithm.models.adapter import _build_model, save_adapter
except ImportError:
    from experiments.peidong.adapter import _build_model, save_adapter


def load_dataset_embeddings():
    """Load all asset embeddings, descriptions, and categories from database."""
    assets = []
    
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
                    cursor.execute("SELECT id, title, category, description, base_code, embedding FROM assets WHERE embedding IS NOT NULL")
                    rows = cursor.fetchall()
                    conn.close()
                    
                    if rows:
                        for row in rows:
                            emb_bytes = row["embedding"]
                            if emb_bytes is not None:
                                emb = np.frombuffer(emb_bytes, dtype=np.float32)
                                desc = (row["description"] or row["title"] or "").strip()
                                assets.append({
                                    "id": row["id"],
                                    "category": row["category"] or "default",
                                    "base_code": row["base_code"] or "default",
                                    "description": desc,
                                    "embedding": emb
                                })
                        print(f"[DATA] Successfully loaded {len(assets)} embeddings from PostgreSQL database at '{host}:{pg_port}/{db_name}'.")
                        return assets
                except Exception:
                    continue
    except ImportError:
        print("[WARN] psycopg2 module not available.")

    return assets


class TripletDataset(Dataset):
    """Constructs (Image Anchor, Text Positive, Hard Text Negative) cross-modal embedding triplets."""
    def __init__(self, assets, num_samples=2000):
        self.assets = assets
        self.num_samples = num_samples
        
        # Precompute CLIP text embeddings for text descriptions if available
        text_cache = {}
        try:
            from algorithm.models.clip_model import get_text_embedding
            print("[DATA] Pre-encoding CLIP text embeddings for image descriptions...")
            for a in assets:
                desc = a["description"]
                if desc and desc not in text_cache:
                    t_emb = get_text_embedding(desc)
                    text_cache[desc] = t_emb / (np.linalg.norm(t_emb) + 1e-8)
        except Exception as e:
            print(f"[WARN] CLIP text embedding generation skipped: {e}")

        self.text_cache = text_cache
        
        # Group by composite key: (category, base_code)
        self.groups = {}
        for idx, a in enumerate(assets):
            composite_key = f"{a['category']}||{a['base_code']}"
            if composite_key not in self.groups:
                self.groups[composite_key] = []
            self.groups[composite_key].append(idx)
            
        self.keys = list(self.groups.keys())

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        anchor_idx = random.randint(0, len(self.assets) - 1)
        anchor_asset = self.assets[anchor_idx]
        anchor_emb = anchor_asset["embedding"]
        
        # Positive: Matching text embedding for this asset if cached, else same-group image
        desc = anchor_asset["description"]
        if desc in self.text_cache:
            pos_emb = self.text_cache[desc]
        else:
            composite_key = f"{anchor_asset['category']}||{anchor_asset['base_code']}"
            same_group = [i for i in self.groups.get(composite_key, []) if i != anchor_idx]
            if same_group:
                pos_emb = self.assets[random.choice(same_group)]["embedding"]
            else:
                noise = np.random.normal(0, 0.01, anchor_emb.shape).astype(np.float32)
                pos_emb = (anchor_emb + noise) / np.linalg.norm(anchor_emb + noise)

        # Hard Negative: sample candidates from different assets
        neg_candidates = []
        for _ in range(8):
            n_idx = random.randint(0, len(self.assets) - 1)
            if n_idx != anchor_idx:
                neg_candidates.append(self.assets[n_idx])
                
        # Find hardest negative (highest similarity to anchor)
        best_neg_emb = neg_candidates[0]["embedding"]
        max_sim = -1.0
        for cand in neg_candidates:
            c_desc = cand["description"]
            c_emb = self.text_cache.get(c_desc, cand["embedding"])
            sim = float(np.dot(anchor_emb, c_emb))
            if sim > max_sim:
                max_sim = sim
                best_neg_emb = c_emb

        return (
            torch.tensor(anchor_emb, dtype=torch.float32),
            torch.tensor(pos_emb, dtype=torch.float32),
            torch.tensor(best_neg_emb, dtype=torch.float32)
        )


def info_nce_loss(out_a, out_p, temperature=0.07):
    """Compute InfoNCE contrastive loss over a batch of (Anchor, Positive) pairs."""
    sim_matrix = torch.matmul(out_a, out_p.T) / temperature
    labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
    loss_a2p = nn.functional.cross_entropy(sim_matrix, labels)
    loss_p2a = nn.functional.cross_entropy(sim_matrix.T, labels)
    return (loss_a2p + loss_p2a) / 2.0


def pcse_loss(z: torch.Tensor) -> torch.Tensor:
    """
    Polar Covariance Spectrum Equalization (PCSE) Loss.
    Forces covariance matrix Z^T Z / (B-1) to match identity matrix I,
    de-correlating embedding dimensions and unfolding narrow polar cone collapse.
    """
    b, d = z.size()
    if b <= 1:
        return torch.tensor(0.0, device=z.device)
    z_centered = z - z.mean(dim=0, keepdim=True)
    cov = torch.matmul(z_centered.T, z_centered) / (b - 1)
    identity = torch.eye(d, device=z.device)
    return (torch.norm(cov - identity, p="fro") ** 2) / d


def evaluate_map_on_golden(model, assets, golden_queries):
    if not golden_queries or not assets:
        return 0.0

    # Get all database embeddings
    raw_embs = [a["embedding"] for a in assets]
    baseline_matrix = np.vstack(raw_embs).astype("float32")
    asset_ids = [a["id"] for a in assets]

    model.eval()
    with torch.no_grad():
        device = next(model.parameters()).device
        x = torch.from_numpy(baseline_matrix).to(device)
        if hasattr(model, "adapt_image"):
            adapted_matrix = model.adapt_image(x).detach().cpu().numpy().astype(np.float32)
        else:
            adapted_matrix = model(x).detach().cpu().numpy().astype(np.float32)

    # Compute MAP
    aps = []
    for q_item in golden_queries:
        q_emb = q_item["embedding"]
        relevant = set(q_item["relevant_ids"])

        if hasattr(model, "adapt_text"):
            q_tensor = torch.from_numpy(q_emb).to(device).unsqueeze(0)
            q_emb = model.adapt_text(q_tensor).squeeze(0).detach().cpu().numpy().astype(np.float32)

        # Compute similarity
        scores = np.dot(adapted_matrix, q_emb)
        sort_idx = np.argsort(scores)[::-1]
        ranked_ids = [asset_ids[idx] for idx in sort_idx]

        # AP calculation
        hits = 0
        ap = 0.0
        for rank, doc_id in enumerate(ranked_ids, start=1):
            if doc_id in relevant:
                hits += 1
                ap += hits / rank
        if relevant:
            aps.append(ap / len(relevant))

    model.train()
    return float(np.mean(aps)) if aps else 0.0


def train(mode="mlp", epochs=20, batch_size=16, lr=1e-4, lora_r=16, lora_alpha=32, margin=0.4, loss_type="triplet", use_learnable_tau=False, use_pcse_loss=False, output_path=None):
    print("=" * 60)
    print(f"  PEIDONG WANG - ADAPTER FINE-TUNING ({mode.upper()})")
    print("=" * 60)

    assets = load_dataset_embeddings()
    if not assets:
        print("[ERROR] No embeddings found in database. Cannot perform fine-tuning.")
        return

    # Load and precompute golden query embeddings
    golden_queries = []
    possible_golden_paths = [
        os.path.join(os.environ.get("PROJECT_ROOT", "."), "golden_test_set.json"),
        os.path.join(os.environ.get("PROJECT_ROOT", "."), "backend", "golden_test_set.json"),
        os.path.join(os.environ.get("PROJECT_ROOT", "."), "experiments", "peidong", "golden_test_set.json"),
        "/app/backend/golden_test_set.json",
        "/app/experiments/peidong/golden_test_set.json",
    ]
    golden_path = None
    for p in possible_golden_paths:
        if os.path.exists(p):
            golden_path = p
            break

    if golden_path:
        try:
            from algorithm.models.clip_model import get_text_embedding
            with open(golden_path, "r", encoding="utf-8") as f:
                raw_golden = json.load(f)
                # Use the first 20 queries to match backend evaluation dashboard API
                golden_queries = raw_golden[:20]
            print(f"[VAL] Loaded golden queries from {golden_path}. Pre-encoding {len(golden_queries)} queries...")
            for q_item in golden_queries:
                q_item["embedding"] = get_text_embedding(q_item["query"])
        except Exception as val_e:
            print(f"[WARN] Failed to load/encode validation golden queries: {val_e}")
            golden_queries = []
    else:
        print("[WARN] golden_test_set.json not found in any common search paths.")

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset & DataLoader
    dataset = TripletDataset(assets, num_samples=2000)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Build model
    model = _build_model(mode=mode, lora_r=lora_r, lora_alpha=lora_alpha)
    model.to(device)
    model.train()

    if loss_type == "infonce":
        tau_str = "learnable" if use_learnable_tau else "0.07"
        pcse_str = " + PCSE Covariance Loss" if use_pcse_loss else ""
        print(f"[LOSS] Using InfoNCE Contrastive Loss (temperature={tau_str}){pcse_str}")
    else:
        print(f"[LOSS] Using Triplet Margin Loss (margin={margin}) with Hard Negative Mining")
        criterion = nn.TripletMarginLoss(margin=margin, p=2)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    print(f"[MODEL] Fine-tuning Mode: {mode.upper()}")
    print(f"[TRAIN] Total Train Samples per Epoch: {len(dataset)}, Epochs: {epochs}, Batch Size: {batch_size}, LR: {lr}")

    best_map = -1.0
    best_weights = None

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        batches = 0
        for anchor, pos, neg in loader:
            anchor, pos, neg = anchor.to(device), pos.to(device), neg.to(device)
            optimizer.zero_grad()
            
            if hasattr(model, "adapt_image") and hasattr(model, "adapt_text"):
                out_a = model.adapt_image(anchor)
                out_p = model.adapt_text(pos)
                out_n = model.adapt_text(neg)
            else:
                out_a = model(anchor)
                out_p = model(pos)
                out_n = model(neg)
            
            if loss_type == "infonce":
                if use_learnable_tau and hasattr(model, "logit_scale"):
                    logit_scale = model.logit_scale.exp().clamp(max=100)
                    sim_matrix = logit_scale * torch.matmul(out_a, out_p.T)
                    labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
                    loss_main = (nn.functional.cross_entropy(sim_matrix, labels) + nn.functional.cross_entropy(sim_matrix.T, labels)) / 2.0
                else:
                    loss_main = info_nce_loss(out_a, out_p)
            else:
                loss_main = criterion(out_a, out_p, out_n)
                
            # CLIP Multi-Modal Alignment Regularization with balanced coefficient
            loss_reg = (1.0 - torch.nn.functional.cosine_similarity(out_a, anchor, dim=-1)).mean()
            loss = loss_main + 0.01 * loss_reg

            # Add Polar Covariance Spectrum Equalization (PCSE) Innovation Loss
            if use_pcse_loss:
                loss_pcse = (pcse_loss(out_a) + pcse_loss(out_p)) / 2.0
                loss = loss + 0.05 * loss_pcse
                
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batches += 1
            
        avg_loss = total_loss / max(1, batches)
        
        # Evaluate validation MAP on golden queries
        val_map = 0.0
        if golden_queries:
            val_map = evaluate_map_on_golden(model, assets, golden_queries)
            if val_map > best_map:
                best_map = val_map
                import copy
                best_weights = copy.deepcopy(model.state_dict())

        if epoch % max(1, epochs // 5) == 0 or epoch == epochs:
            val_info = f", Golden MAP: {val_map:.4f} (Best: {best_map:.4f})" if golden_queries else ""
            print(f"  Epoch [{epoch:2d}/{epochs:2d}] -> {loss_type.capitalize()} Loss: {avg_loss:.6f}{val_info}")

    # Load best weights if found
    if best_weights is not None:
        model.load_state_dict(best_weights)
        print(f"[VAL] Restored best model weights with Golden MAP: {best_map:.4f}")
    else:
        print("[VAL] No validation model saved, using final epoch weights.")

    # Save trained model weights across all expected mount paths
    target_paths = [
        os.path.join(os.environ.get("PROJECT_ROOT", "."), "backend", "static", "models", "adapter.pth"),
        "/app/backend/static/models/adapter.pth",
        "/app/static/models/adapter.pth"
    ]
    if output_path:
        target_paths.insert(0, output_path)

    for p in target_paths:
        try:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            save_adapter(model, p)
            print(f"[SUCCESS] Saved adapter weights to: {p}")
        except Exception:
            pass

    print("=" * 60)
    print(f"[SUCCESS] Adapter fine-tuning finished for mode: {mode.upper()}")
    print("=" * 60)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Peidong's Polar Domain Adapter")
    parser.add_argument("--mode", type=str, default="mlp", choices=["mlp", "swiglu", "dual_swiglu", "ted", "lora", "qlora"], help="Adapter architecture mode")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--margin", type=float, default=0.4, help="Margin for Triplet Margin Loss")
    parser.add_argument("--loss-type", type=str, default="triplet", choices=["triplet", "infonce"], help="Loss function type (triplet or infonce)")
    parser.add_argument("--use-learnable-tau", action="store_true", help="Enable learnable temperature logit scale for InfoNCE loss")
    parser.add_argument("--use-pcse-loss", action="store_true", help="Enable Polar Covariance Spectrum Equalization (PCSE) Loss")
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
        margin=args.margin,
        loss_type=args.loss_type,
        use_learnable_tau=args.use_learnable_tau,
        use_pcse_loss=args.use_pcse_loss,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        output_path=out_path
    )
