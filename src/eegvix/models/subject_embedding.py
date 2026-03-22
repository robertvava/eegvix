"""Per-subject learnable embedding for multi-subject EEG encoding."""

import torch
import torch.nn as nn


class SubjectEmbedding(nn.Module):
    """Learnable embedding vector per subject, added to the EEG encoder's CLS token.

    During a warmup phase early in training, embeddings can be zeroed out
    to let the encoder learn subject-invariant features first.
    """

    def __init__(self, n_subjects: int = 10, embed_dim: int = 512):
        super().__init__()
        self.embedding = nn.Embedding(n_subjects, embed_dim)
        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(self, subject_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            subject_ids: (batch,) integer tensor with subject indices

        Returns:
            (batch, embed_dim) subject embedding vectors
        """
        return self.embedding(subject_ids)
