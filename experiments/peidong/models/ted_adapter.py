"""
experiments/peidong/models/ted_adapter.py
TED-Adapter (Temporal-Entity Disentangled Subspace Co-Adapter Subclass).
"""
import torch
import torch.nn as nn
from experiments.peidong.models.base_adapter import BaseAdapter


class TEDAdapter(BaseAdapter):
    """
    TED-Adapter: Disentangles 512d representation space into 256d Entity Invariant Subspace
    and 256d Temporal Variation Subspace.
    """
    def __init__(self, input_dim: int = 512, hidden_dim: int = 1024, output_dim: int = 512):
        super().__init__(input_dim=input_dim, output_dim=output_dim, mode_name="ted")
        self.half_dim = output_dim // 2

        # Entity Invariant Branch (256d)
        self.img_ent_ln = nn.LayerNorm(input_dim)
        self.img_ent_w1 = nn.Linear(input_dim, hidden_dim)
        self.img_ent_w2 = nn.Linear(input_dim, hidden_dim)
        self.img_ent_w3 = nn.Linear(hidden_dim, self.half_dim)

        self.txt_ent_ln = nn.LayerNorm(input_dim)
        self.txt_ent_w1 = nn.Linear(input_dim, hidden_dim)
        self.txt_ent_w2 = nn.Linear(input_dim, hidden_dim)
        self.txt_ent_w3 = nn.Linear(hidden_dim, self.half_dim)

        # Temporal Variation Branch (256d)
        self.img_temp_ln = nn.LayerNorm(input_dim)
        self.img_temp_w1 = nn.Linear(input_dim, hidden_dim)
        self.img_temp_w2 = nn.Linear(input_dim, hidden_dim)
        self.img_temp_w3 = nn.Linear(hidden_dim, self.half_dim)

        self.txt_temp_ln = nn.LayerNorm(input_dim)
        self.txt_temp_w1 = nn.Linear(input_dim, hidden_dim)
        self.txt_temp_w2 = nn.Linear(input_dim, hidden_dim)
        self.txt_temp_w3 = nn.Linear(hidden_dim, self.half_dim)

    def adapt_image(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.img_ent_ln(x)
        g1 = nn.functional.silu(self.img_ent_w1(h1)) * self.img_ent_w2(h1)
        ent_part = x[:, :self.half_dim] + self.gamma_img * self.img_ent_w3(g1)

        h2 = self.img_temp_ln(x)
        g2 = nn.functional.silu(self.img_temp_w1(h2)) * self.img_temp_w2(h2)
        temp_part = x[:, self.half_dim:] + self.gamma_img * self.img_temp_w3(g2)

        out = torch.cat([ent_part, temp_part], dim=-1)
        return nn.functional.normalize(out, p=2, dim=-1)

    def adapt_text(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.txt_ent_ln(x)
        g1 = nn.functional.silu(self.txt_ent_w1(h1)) * self.txt_ent_w2(h1)
        ent_part = x[:, :self.half_dim] + self.gamma_txt * self.txt_ent_w3(g1)

        h2 = self.txt_temp_ln(x)
        g2 = nn.functional.silu(self.txt_temp_w1(h2)) * self.txt_temp_w2(h2)
        temp_part = x[:, self.half_dim:] + self.gamma_txt * self.txt_temp_w3(g2)

        out = torch.cat([ent_part, temp_part], dim=-1)
        return nn.functional.normalize(out, p=2, dim=-1)
