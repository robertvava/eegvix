"""Retrieval accuracy and zero-shot identification metrics."""

import torch
import numpy as np


def top_k_accuracy(
    eeg_embeds: torch.Tensor,
    image_embeds: torch.Tensor,
    k_values: list[int] | None = None,
) -> dict[str, float]:
    """Compute top-k retrieval accuracy: for each EEG, check if the correct image is in top-k.

    Args:
        eeg_embeds: (n, d) L2-normalized EEG embeddings
        image_embeds: (n, d) L2-normalized image embeddings (same ordering as eeg_embeds)
        k_values: List of k values to evaluate

    Returns:
        Dict mapping "top_k" to accuracy for each k
    """
    if k_values is None:
        k_values = [1, 5, 10, 50, 200]

    similarity = eeg_embeds @ image_embeds.T  # (n, n)
    labels = torch.arange(similarity.size(0), device=similarity.device)

    results = {}
    for k in k_values:
        if k > similarity.size(1):
            continue
        top_k_preds = similarity.topk(k, dim=1).indices
        correct = (top_k_preds == labels.unsqueeze(1)).any(dim=1).float().mean()
        results[f"top_{k}"] = correct.item()

    return results


def zero_shot_identification(
    bio_test_embeds: torch.Tensor,
    syn_test_embeds: torch.Tensor,
    distractor_embeds: torch.Tensor | None = None,
) -> dict[str, float]:
    """Zero-shot identification following the THINGS-EEG2 paper protocol.

    For each biological test EEG embedding, check if the correlation with its
    matching synthetic embedding is higher than with all other candidates.

    Args:
        bio_test_embeds: (200, d) biological test EEG embeddings (averaged across 80 reps)
        syn_test_embeds: (200, d) synthetic test embeddings (from CLIP image encoder)
        distractor_embeds: (n_distractors, d) optional distractor embeddings

    Returns:
        Dict with identification accuracy and per-condition results
    """
    n_test = bio_test_embeds.size(0)

    # Build candidate pool
    if distractor_embeds is not None:
        candidates = torch.cat([syn_test_embeds, distractor_embeds], dim=0)
    else:
        candidates = syn_test_embeds

    # Correlation matrix
    similarity = bio_test_embeds @ candidates.T  # (200, n_candidates)

    # For each test condition, check if the correct candidate has the highest similarity
    correct = 0
    for i in range(n_test):
        # The correct match is at index i in candidates (first 200 are syn_test)
        if similarity[i].argmax().item() == i:
            correct += 1

    accuracy = correct / n_test

    return {
        "accuracy": accuracy,
        "n_correct": correct,
        "n_total": n_test,
        "n_candidates": candidates.size(0),
    }


def zero_shot_with_varying_distractors(
    bio_test_embeds: torch.Tensor,
    syn_test_embeds: torch.Tensor,
    distractor_embeds: torch.Tensor,
    set_sizes: list[int] | None = None,
    n_iterations: int = 100,
) -> dict[int, float]:
    """Run zero-shot identification with varying numbers of distractors.

    Replicates the paper's protocol of gradually increasing candidate set size.

    Args:
        bio_test_embeds: (200, d) biological test embeddings
        syn_test_embeds: (200, d) synthetic test embeddings
        distractor_embeds: (n, d) pool of distractor embeddings
        set_sizes: List of distractor set sizes to evaluate
        n_iterations: Number of random iterations per set size

    Returns:
        Dict mapping set_size to mean accuracy
    """
    if set_sizes is None:
        set_sizes = list(range(0, min(distractor_embeds.size(0), 150001), 1000))

    results = {}
    n_distractors = distractor_embeds.size(0)

    for size in set_sizes:
        accuracies = []
        for _ in range(n_iterations):
            if size == 0:
                result = zero_shot_identification(bio_test_embeds, syn_test_embeds)
            else:
                idx = torch.randperm(n_distractors)[:size]
                distractors = distractor_embeds[idx]
                result = zero_shot_identification(bio_test_embeds, syn_test_embeds, distractors)
            accuracies.append(result["accuracy"])

        results[size] = float(np.mean(accuracies))

    return results
