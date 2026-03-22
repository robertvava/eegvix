"""Tests for evaluation metrics."""

import torch
from eegvix.evaluation.retrieval import top_k_accuracy, zero_shot_identification


class TestTopKAccuracy:
    def test_perfect_match(self):
        """When embeddings are identical, top-1 accuracy should be 1.0."""
        embeds = torch.nn.functional.normalize(torch.randn(20, 768), dim=-1)
        results = top_k_accuracy(embeds, embeds, k_values=[1, 5])
        assert results["top_1"] == 1.0
        assert results["top_5"] == 1.0

    def test_random_chance(self):
        """With random embeddings, top-1 accuracy should be near 1/n."""
        torch.manual_seed(42)
        n = 200
        eeg = torch.nn.functional.normalize(torch.randn(n, 768), dim=-1)
        img = torch.nn.functional.normalize(torch.randn(n, 768), dim=-1)
        results = top_k_accuracy(eeg, img, k_values=[1])
        # Should be near chance (0.5%) but with some variance
        assert results["top_1"] < 0.1


class TestZeroShotIdentification:
    def test_perfect_identification(self):
        """With identical embeddings, all conditions should be correctly identified."""
        embeds = torch.nn.functional.normalize(torch.randn(50, 768), dim=-1)
        result = zero_shot_identification(embeds, embeds)
        assert result["accuracy"] == 1.0
        assert result["n_correct"] == 50

    def test_with_distractors(self):
        """Identification should still work with well-separated embeddings + distractors."""
        torch.manual_seed(42)
        embeds = torch.nn.functional.normalize(torch.randn(10, 768), dim=-1)
        distractors = torch.nn.functional.normalize(torch.randn(100, 768), dim=-1)
        result = zero_shot_identification(embeds, embeds, distractors)
        # With identical bio/syn and random distractors, most should be correct
        assert result["accuracy"] >= 0.5
        assert result["n_candidates"] == 110
