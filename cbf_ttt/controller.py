"""A small regressor for one shared, continuous chunk forgetting coefficient."""

import torch
from torch import nn


class ForgettingController(nn.Module):
    def __init__(self, hidden_size: int, scalar_size: int, width: int = 128, semantic_size: int = 64):
        super().__init__()
        self.hidden_size = hidden_size
        self.scalar_size = scalar_size
        self.width = width
        self.semantic_size = semantic_size
        self.semantic = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, semantic_size), nn.GELU())
        self.regressor = nn.Sequential(
            nn.Linear(semantic_size + scalar_size, width), nn.GELU(),
            nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1),
        )
        self.register_buffer("scalar_mean", torch.zeros(scalar_size))
        self.register_buffer("scalar_std", torch.ones(scalar_size))

    def forward(self, semantic: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        """Return the unprojected score so MSE keeps its gradient outside [0, 1]."""
        semantic = semantic.float()
        scalars = scalars.float()
        standardized = (scalars - self.scalar_mean) / self.scalar_std.clamp_min(1e-6)
        return self.regressor(torch.cat((self.semantic(semantic), standardized), dim=-1)).squeeze(-1)

    def predict(self, semantic: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        return self.forward(semantic, scalars).clamp(0.0, 1.0)

    def set_statistics(self, scalars: torch.Tensor) -> None:
        if scalars.ndim != 2 or scalars.shape[1] != self.scalar_size:
            raise ValueError("scalar statistics have the wrong shape")
        self.scalar_mean.copy_(scalars.float().mean(0))
        self.scalar_std.copy_(scalars.float().std(0, unbiased=False).clamp_min(1e-6))
