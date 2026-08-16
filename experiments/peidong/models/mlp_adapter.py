"""
experiments/peidong/models/mlp_adapter.py
2-Layer MLP Projection Adapter Subclass.
"""
import torch
import torch.nn as nn
from experiments.peidong.models.base_adapter import BaseAdapter


class MLPAdapter(BaseAdapter):
    """2-Layer MLP Projection Adapter."""
    def __init__(self, input_dim: int = 512, hidden_dim: int = 1024, output_dim: int = 512):
        super().__init__(input_dim=input_dim, output_dim=output_dim, mode_name="mlp")
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        nn.init.kaiming_uniform_(self.net[0].weight)
        nn.init.zeros_(self.net[0].bias)
        nn.init.normal_(self.net[2].weight, std=0.01)
        nn.init.zeros_(self.net[2].bias)

    def adapt_image(self, x: torch.Tensor) -> torch.Tensor:
        out = x + 0.1 * self.net(x)
        return nn.functional.normalize(out, p=2, dim=-1)

    def adapt_text(self, x: torch.Tensor) -> torch.Tensor:
        return x
