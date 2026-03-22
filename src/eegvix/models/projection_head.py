"""Projection head for contrastive learning.

Maps encoder output to a normalized embedding space where the contrastive loss operates.
Standard practice in CLIP/SimCLR: the projection head separates the representation space
(used for downstream tasks) from the contrastive loss space.
"""

import torch
import torch.nn as nn


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int = 768, hidden_dim: int = 2048, output_dim: int = 768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project and L2-normalize.

        Args:
            x: (batch, input_dim)

        Returns:
            (batch, output_dim) L2-normalized embeddings
        """
        projected = self.net(x)
        return nn.functional.normalize(projected, dim=-1)
