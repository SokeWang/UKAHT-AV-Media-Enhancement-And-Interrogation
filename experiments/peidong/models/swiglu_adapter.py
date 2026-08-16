"""
experiments/peidong/models/swiglu_adapter.py
Single-Branch and Dual-Branch SwiGLU Gated Residual Co-Adapter Subclasses.
"""
import torch
import torch.nn as nn
from experiments.peidong.models.base_adapter import BaseAdapter


class SingleSwiGLUAdapter(BaseAdapter):
    """Single-Branch Image-Only SwiGLU Gated Residual Adapter."""
    def __init__(self, input_dim: int = 512, hidden_dim: int = 1024, output_dim: int = 512):
        super().__init__(input_dim=input_dim, output_dim=output_dim, mode_name="swiglu")
        self.ln = nn.LayerNorm(input_dim)
        self.w1 = nn.Linear(input_dim, hidden_dim)
        self.w2 = nn.Linear(input_dim, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, output_dim)
        nn.init.kaiming_uniform_(self.w1.weight)
        nn.init.kaiming_uniform_(self.w2.weight)
        nn.init.zeros_(self.w3.weight)
        nn.init.zeros_(self.w3.bias)

    def adapt_image(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln(x)
        gate = nn.functional.silu(self.w1(h)) * self.w2(h)
        out = x + self.gamma_img * self.w3(gate)
        return nn.functional.normalize(out, p=2, dim=-1)

    def adapt_text(self, x: torch.Tensor) -> torch.Tensor:
        return x


class DualSwiGLUAdapter(BaseAdapter):
    """Dual-Branch Image & Text SwiGLU Gated Residual Co-Adapter."""
    def __init__(self, input_dim: int = 512, hidden_dim: int = 1024, output_dim: int = 512):
        super().__init__(input_dim=input_dim, output_dim=output_dim, mode_name="dual_swiglu")
        # Image branch
        self.img_ln = nn.LayerNorm(input_dim)
        self.img_w1 = nn.Linear(input_dim, hidden_dim)
        self.img_w2 = nn.Linear(input_dim, hidden_dim)
        self.img_w3 = nn.Linear(hidden_dim, output_dim)
        nn.init.kaiming_uniform_(self.img_w1.weight)
        nn.init.kaiming_uniform_(self.img_w2.weight)
        nn.init.zeros_(self.img_w3.weight)
        nn.init.zeros_(self.img_w3.bias)

        # Text branch
        self.txt_ln = nn.LayerNorm(input_dim)
        self.txt_w1 = nn.Linear(input_dim, hidden_dim)
        self.txt_w2 = nn.Linear(input_dim, hidden_dim)
        self.txt_w3 = nn.Linear(hidden_dim, output_dim)
        nn.init.kaiming_uniform_(self.txt_w1.weight)
        nn.init.kaiming_uniform_(self.txt_w2.weight)
        nn.init.zeros_(self.txt_w3.weight)
        nn.init.zeros_(self.txt_w3.bias)

    def adapt_image(self, x: torch.Tensor) -> torch.Tensor:
        h = self.img_ln(x)
        gate = nn.functional.silu(self.img_w1(h)) * self.img_w2(h)
        out = x + self.gamma_img * self.img_w3(gate)
        return nn.functional.normalize(out, p=2, dim=-1)

    def adapt_text(self, x: torch.Tensor) -> torch.Tensor:
        h = self.txt_ln(x)
        gate = nn.functional.silu(self.txt_w1(h)) * self.txt_w2(h)
        out = x + self.gamma_txt * self.txt_w3(gate)
        return nn.functional.normalize(out, p=2, dim=-1)
