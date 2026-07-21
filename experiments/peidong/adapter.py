"""
experiments/peidong/adapter.py
Owner: Peidong Wang — Milestone 3 (polar domain MLP/LoRA/QLoRA projection adapter)

Responsibilities:
  - Define the AdapterModel with configurable modes: 'mlp', 'lora', 'qlora'.
  - Provide load/save helpers so the trained weights can be dropped in.
"""

import os
import numpy as np

# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

def _build_model(input_dim: int = 512, hidden_dim: int = 1024, output_dim: int = 512, mode: str = "mlp", lora_r: int = 16, lora_alpha: int = 32):
    """
    Build an adapter model supporting multiple fine-tuning configurations:
    - MLP: 2-layer MLP (Linear -> ReLU -> Linear).
    - LoRA: Low-Rank Adapter mapping (Identity base + FP32 low-rank projection).
    - QLoRA: NF4 Quantized Low-Rank Adapter.
    """
    import torch
    import torch.nn as nn
    import math

    class AdapterModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.mode = mode.lower()
            self.input_dim = input_dim
            self.output_dim = output_dim
            self.lora_r = lora_r
            self.lora_alpha = lora_alpha
            
            if self.mode == "mlp":
                self.net = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, output_dim)
                )
            elif self.mode in ("lora", "qlora"):
                # Base weight initialized as identity projection (frozen)
                self.base_weight = nn.Parameter(torch.eye(output_dim, input_dim), requires_grad=False)
                
                # LoRA trainable paths
                self.lora_A = nn.Linear(input_dim, lora_r, bias=False)
                self.lora_B = nn.Linear(lora_r, output_dim, bias=False)
                self.scaling = lora_alpha / lora_r
                
                # Standard LoRA initialization: A is Gaussian, B is Zero
                nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
                nn.init.zeros_(self.lora_B.weight)
                
                if self.mode == "qlora":
                    # Register NF4 quantization levels
                    self.register_buffer("nf4_levels", torch.tensor([
                        -1.0, -0.6961928, -0.5250716, -0.3949184,
                        -0.2844413, -0.1847734, -0.0910502, 0.0,
                        0.0795803, 0.1609302, 0.2461159, 0.3379152,
                        0.4407079, 0.562617, 0.7229568, 1.0
                    ]))
                    # Compute scale and quantize base weight
                    scale = self.base_weight.abs().max()
                    if scale == 0:
                        scale = 1.0
                    self.register_buffer("weight_scale", torch.tensor(float(scale)))
                    
                    normalized = self.base_weight / scale
                    diffs = (normalized.unsqueeze(-1) - self.nf4_levels.to(self.base_weight.device)).abs()
                    indices = diffs.argmin(dim=-1)
                    self.register_buffer("quantized_indices", indices)
            else:
                raise ValueError(f"Unknown mode: {mode}")

        def _dequantize_nf4(self):
            levels = self.nf4_levels.to(self.quantized_indices.device)
            dequantized = levels[self.quantized_indices] * self.weight_scale
            return dequantized

        def forward(self, x):
            if self.mode == "mlp":
                out = self.net(x)
            elif self.mode == "lora":
                base_out = x @ self.base_weight.t()
                lora_out = self.lora_B(self.lora_A(x)) * self.scaling
                out = base_out + lora_out
            elif self.mode == "qlora":
                dequantized_weight = self._dequantize_nf4()
                base_out = x @ dequantized_weight.t()
                lora_out = self.lora_B(self.lora_A(x)) * self.scaling
                out = base_out + lora_out
                
            # L2 normalise
            return out / (out.norm(dim=-1, keepdim=True) + 1e-8)

    return AdapterModel()


# ---------------------------------------------------------------------------
# Load / save helpers
# ---------------------------------------------------------------------------

def load_adapter(weights_path: str, mode: str = "mlp", lora_r: int = 16, lora_alpha: int = 32):
    """
    Load a trained AdapterModel from a .pth file.
    Supports auto-detecting the architecture metadata from checkpoint or falling back to defaults.
    """
    import torch

    if not os.path.exists(weights_path):
        return None

    try:
        checkpoint = torch.load(weights_path, map_location="cpu")
        
        # Check if saved with metadata dictionary
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            loaded_mode = checkpoint.get("mode", mode)
            input_dim = checkpoint.get("input_dim", 512)
            output_dim = checkpoint.get("output_dim", 512)
            loaded_r = checkpoint.get("lora_r", lora_r)
            loaded_alpha = checkpoint.get("lora_alpha", lora_alpha)
            
            model = _build_model(
                input_dim=input_dim,
                output_dim=output_dim,
                mode=loaded_mode,
                lora_r=loaded_r,
                lora_alpha=loaded_alpha
            )
            model.load_state_dict(checkpoint["state_dict"])
        else:
            # Backward compatibility check for raw state dict
            model = _build_model(mode=mode, lora_r=lora_r, lora_alpha=lora_alpha)
            model.load_state_dict(checkpoint)
            
        model.eval()
        return model
    except Exception as e:
        print(f"[WARN] Failed to load adapter from {weights_path}: {e}")
        return None


def save_adapter(model, weights_path: str) -> None:
    """Persist adapter weights alongside architecture metadata for automatic reloading."""
    import torch
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)
    
    checkpoint = {
        "state_dict": model.state_dict(),
        "mode": getattr(model, "mode", "mlp"),
        "input_dim": getattr(model, "input_dim", 512),
        "output_dim": getattr(model, "output_dim", 512),
        "lora_r": getattr(model, "lora_r", 16),
        "lora_alpha": getattr(model, "lora_alpha", 32),
    }
    torch.save(checkpoint, weights_path)


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

def apply_adapter(embedding: np.ndarray, adapter) -> np.ndarray:
    """
    Pass CLIP embeddings through the trained adapter.

    Args:
        embedding: CLIP embedding(s) as a float32 numpy array. Can be 1D (512,) or 2D (N, 512).
        adapter:   AdapterModel instance returned by load_adapter(), or None.

    Returns:
        Adapted float32 numpy array of same shape. Falls back to original if adapter is None.
    """
    if adapter is None:
        return embedding

    import torch

    is_1d = (embedding.ndim == 1)
    if is_1d:
        x = torch.from_numpy(embedding).unsqueeze(0)
    else:
        x = torch.from_numpy(embedding)

    with torch.no_grad():
        out = adapter(x)

    if is_1d:
        return out.squeeze(0).numpy().astype(np.float32)
    return out.numpy().astype(np.float32)
