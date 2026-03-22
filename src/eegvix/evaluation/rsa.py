"""Representational Similarity Analysis (RSA) for comparing EEG and image representations."""

import torch
import numpy as np
from scipy.stats import spearmanr


def compute_rdm(embeddings: torch.Tensor) -> torch.Tensor:
    """Compute Representational Dissimilarity Matrix (1 - cosine similarity).

    Args:
        embeddings: (n, d) L2-normalized embeddings

    Returns:
        (n, n) dissimilarity matrix
    """
    similarity = embeddings @ embeddings.T
    return 1.0 - similarity


def rsa_correlation(
    eeg_embeds: torch.Tensor,
    image_embeds: torch.Tensor,
) -> dict[str, float]:
    """Compute RSA between EEG and image representational spaces.

    Compares the geometry of the two embedding spaces by correlating their
    representational dissimilarity matrices (RDMs).

    Args:
        eeg_embeds: (n, d) L2-normalized EEG embeddings
        image_embeds: (n, d) L2-normalized image embeddings

    Returns:
        Dict with Pearson and Spearman correlations between RDMs
    """
    eeg_rdm = compute_rdm(eeg_embeds).cpu().numpy()
    img_rdm = compute_rdm(image_embeds).cpu().numpy()

    # Extract upper triangle (excluding diagonal)
    n = eeg_rdm.shape[0]
    triu_idx = np.triu_indices(n, k=1)
    eeg_upper = eeg_rdm[triu_idx]
    img_upper = img_rdm[triu_idx]

    pearson_r = float(np.corrcoef(eeg_upper, img_upper)[0, 1])
    spearman_r, spearman_p = spearmanr(eeg_upper, img_upper)

    return {
        "pearson_r": pearson_r,
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
    }
