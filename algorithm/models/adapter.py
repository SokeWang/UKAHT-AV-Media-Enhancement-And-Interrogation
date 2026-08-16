"""
algorithm/models/adapter.py
Clean re-export bridge for Peidong's Modular Polar Adapters.
"""
import sys, os
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.peidong.models import (
    BaseAdapter,
    MLPAdapter,
    SingleSwiGLUAdapter,
    DualSwiGLUAdapter,
    PCSEAdapter,
    TEDAdapter,
    build_adapter,
    load_adapter as _load_adapter,
)


def _build_model(input_dim: int = 512, hidden_dim: int = 1024, output_dim: int = 512, mode: str = "mlp", **kwargs):
    """Backward-compatible wrapper for model building."""
    return build_adapter(mode=mode, input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)


def save_adapter(model, weights_path: str) -> None:
    """Save adapter model using its persisted method or fallback."""
    if hasattr(model, "save"):
        model.save(weights_path)
    else:
        import torch
        os.makedirs(os.path.dirname(weights_path), exist_ok=True)
        torch.save(model.state_dict(), weights_path)


def load_adapter(weights_path: str, mode: str = "mlp", **kwargs):
    """Load adapter model using polymorphic factory."""
    return _load_adapter(weights_path=weights_path, mode=mode)


def apply_adapter(embedding: np.ndarray, adapter) -> np.ndarray:
    """Pass CLIP embeddings through trained adapter for inference."""
    if adapter is None:
        return embedding

    import torch

    is_1d = (embedding.ndim == 1)
    x = torch.from_numpy(embedding).unsqueeze(0) if is_1d else torch.from_numpy(embedding)

    adapter.eval()
    with torch.no_grad():
        if hasattr(adapter, "adapt_image"):
            out = adapter.adapt_image(x)
        else:
            out = adapter(x)

    out_np = out.detach().cpu().numpy().astype(np.float32)
    return out_np.squeeze(0) if is_1d else out_np
