"""
experiments/peidong/models/pcse_adapter.py
PCSE-ACRA (Polar Covariance Spectrum Equalization & Anisotropy-Corrected Residual Adapter).
"""
import torch
from experiments.peidong.models.swiglu_adapter import DualSwiGLUAdapter


class PCSEAdapter(DualSwiGLUAdapter):
    """PCSE-ACRA Framework subclass."""
    def __init__(self, input_dim: int = 512, hidden_dim: int = 1024, output_dim: int = 512):
        super().__init__(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
        self.mode_name = "pcse"

    @staticmethod
    def compute_pcse_loss(z: torch.Tensor) -> torch.Tensor:
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
