"""
experiments/peidong/models/base_adapter.py
Abstract Base Class for all Polar Domain CLIP Adapters.
"""
import os
import torch
import torch.nn as nn
import numpy as np


class BaseAdapter(nn.Module):
    """
    Abstract Base Adapter class enforcing unified inference and persistence APIs.
    """
    def __init__(self, input_dim: int = 512, output_dim: int = 512, mode_name: str = "base"):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mode_name = mode_name
        self.gamma_img = nn.Parameter(torch.tensor(0.35, dtype=torch.float32))
        self.gamma_txt = nn.Parameter(torch.tensor(0.35, dtype=torch.float32))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def adapt_image(self, x: torch.Tensor) -> torch.Tensor:
        """Project image embedding tensor into aligned polar manifold."""
        raise NotImplementedError

    def adapt_text(self, x: torch.Tensor) -> torch.Tensor:
        """Project query text embedding tensor into aligned polar manifold."""
        return x

    def forward(self, x: torch.Tensor, modality: str = "image") -> torch.Tensor:
        """Unified forward pass fallback."""
        if modality == "text":
            return self.adapt_text(x)
        return self.adapt_image(x)

    def save(self, weights_path: str) -> None:
        """Save model state dict alongside architecture metadata."""
        os.makedirs(os.path.dirname(weights_path), exist_ok=True)
        checkpoint = {
            "state_dict": self.state_dict(),
            "mode": self.mode_name,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
        }
        torch.save(checkpoint, weights_path)
