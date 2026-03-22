"""Tests for contrastive loss."""

import pytest
import torch

from eegvix.losses.contrastive import InfoNCELoss


class TestInfoNCELoss:
    def test_perfect_alignment(self):
        """When EEG and image embeddings are identical, accuracy should be 1.0."""
        loss_fn = InfoNCELoss(init_temperature=0.07, learnable=False)
        embeds = torch.nn.functional.normalize(torch.randn(8, 768), dim=-1)
        result = loss_fn(embeds, embeds)
        # Loss won't be near zero because cross-entropy with cosine ~1/temp is still nonzero,
        # but accuracy must be perfect since the diagonal is the max
        assert result["eeg_to_img_acc"].item() == 1.0
        assert result["img_to_eeg_acc"].item() == 1.0

    def test_random_embeddings(self):
        """With random embeddings, accuracy should be near chance (1/batch_size)."""
        loss_fn = InfoNCELoss(init_temperature=0.07, learnable=False)
        eeg = torch.nn.functional.normalize(torch.randn(64, 768), dim=-1)
        img = torch.nn.functional.normalize(torch.randn(64, 768), dim=-1)
        result = loss_fn(eeg, img)
        # Loss should be high (close to log(64) ≈ 4.16)
        assert result["loss"].item() > 3.0

    def test_gradient_flows_to_temperature(self):
        loss_fn = InfoNCELoss(init_temperature=0.07, learnable=True)
        eeg = torch.nn.functional.normalize(torch.randn(8, 768), dim=-1)
        img = torch.nn.functional.normalize(torch.randn(8, 768), dim=-1)
        result = loss_fn(eeg, img)
        result["loss"].backward()
        assert loss_fn.log_temperature.grad is not None

    def test_symmetric(self):
        """Loss should be symmetric: L(a, b) == L(b, a)."""
        loss_fn = InfoNCELoss(init_temperature=0.07, learnable=False)
        eeg = torch.nn.functional.normalize(torch.randn(8, 768), dim=-1)
        img = torch.nn.functional.normalize(torch.randn(8, 768), dim=-1)
        r1 = loss_fn(eeg, img)
        r2 = loss_fn(img, eeg)
        assert torch.allclose(r1["loss"], r2["loss"], atol=1e-5)

    def test_batch_size_1(self):
        """Should not crash with batch size 1."""
        loss_fn = InfoNCELoss(init_temperature=0.07, learnable=False)
        eeg = torch.nn.functional.normalize(torch.randn(1, 768), dim=-1)
        img = torch.nn.functional.normalize(torch.randn(1, 768), dim=-1)
        result = loss_fn(eeg, img)
        assert not torch.isnan(result["loss"])
