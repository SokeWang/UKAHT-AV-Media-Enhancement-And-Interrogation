"""
backend/models/adapter.py
Owner: Peidong Wang — Milestone 3 (polar domain MLP projection adapter)

Responsibilities:
  - Define the AdapterModel (2-layer MLP) that re-projects CLIP embeddings
    into a polar-domain-aligned vector space using Triplet Loss training.
  - Provide load/save helpers so the trained weights can be dropped in
    without changing retrieval/search.py.
  - Training is done separately in notebooks/train_adapter.ipynb.
    This module is the inference-time API.

Usage at inference time:
    from backend.models.adapter import load_adapter, apply_adapter
    adapter = load_adapter("static/models/adapter.pth")
    adapted_emb = apply_adapter(clip_embedding, adapter)
"""

import os
import numpy as np

# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

def _build_model(input_dim: int = 512, hidden_dim: int = 1024, output_dim: int = 512):
    """
    Build a 2-layer MLP adapter:
        Linear(512 → 1024) → ReLU → Linear(1024 → 512) → L2-normalise

    Architecture details are intentionally left adjustable here so that
    the training notebook can experiment with sizes before freezing them.
    """
    import torch.nn as nn

    class AdapterModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
            )

        def forward(self, x):
            out = self.net(x)
            # L2 normalise so output lives on the unit hypersphere
            return out / (out.norm(dim=-1, keepdim=True) + 1e-8)

    return AdapterModel()


# ---------------------------------------------------------------------------
# Load / save helpers
# ---------------------------------------------------------------------------

def load_adapter(weights_path: str):
    """
    Load a trained AdapterModel from a .pth file.

    Returns the model in eval mode, or None if the file does not exist yet
    (so the rest of the system degrades gracefully to vanilla CLIP).
    """
    import torch

    if not os.path.exists(weights_path):
        return None

    model = _build_model()
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    return model


def save_adapter(model, weights_path: str) -> None:
    """Persist adapter weights — called from the training notebook."""
    import torch
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)
    torch.save(model.state_dict(), weights_path)


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

def apply_adapter(embedding: np.ndarray, adapter) -> np.ndarray:
    """
    Pass a CLIP embedding through the trained adapter.

    Args:
        embedding: 512-dim float32 numpy array (already L2-normalised).
        adapter:   AdapterModel instance returned by load_adapter(), or None.

    Returns:
        Adapted 512-dim float32 numpy array.
        Falls back to the original embedding if adapter is None.
    """
    if adapter is None:
        return embedding

    import torch

    x = torch.from_numpy(embedding).unsqueeze(0)
    with torch.no_grad():
        out = adapter(x)
    return out.squeeze(0).numpy().astype(np.float32)
