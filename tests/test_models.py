"""Tests for EEG encoder, projection head, and subject embedding."""

import pytest
import torch

from eegvix.models.eeg_encoder import EEGEncoder, TemporalConvStem, SpatialChannelAttention, FrequencyBranch
from eegvix.models.projection_head import ProjectionHead
from eegvix.models.subject_embedding import SubjectEmbedding


class TestTemporalConvStem:
    def test_output_shape(self):
        stem = TemporalConvStem(in_channels=1, embed_dim=256, kernel_sizes=[7, 5])
        x = torch.randn(17, 1, 100)  # 17 channels, 1 input feature, 100 timepoints
        out = stem(x)
        assert out.shape[0] == 17
        assert out.shape[1] == 256
        assert out.shape[2] > 0  # Reduced temporal dimension
        assert out.shape[2] < 100


class TestSpatialChannelAttention:
    def test_output_shape(self):
        attn = SpatialChannelAttention(embed_dim=256, n_heads=4)
        x = torch.randn(2, 17, 256)  # batch=2, 17 channels, embed_dim=256
        out = attn(x)
        assert out.shape == (2, 17, 256)


class TestFrequencyBranch:
    def test_output_shape(self):
        branch = FrequencyBranch(n_channels=17, n_timepoints=100, output_dim=128)
        eeg = torch.randn(2, 17, 100)
        out = branch(eeg)
        assert out.shape == (2, 128)


class TestEEGEncoder:
    def test_output_shape(self, dummy_eeg, dummy_subject_ids, output_dim):
        encoder = EEGEncoder(output_dim=output_dim, embed_dim=128, n_temporal_transformer_layers=2)
        out = encoder(dummy_eeg, dummy_subject_ids)
        assert out.shape == (dummy_eeg.size(0), output_dim)

    def test_gradient_flow(self, dummy_eeg, dummy_subject_ids):
        encoder = EEGEncoder(embed_dim=128, n_temporal_transformer_layers=2)
        out = encoder(dummy_eeg, dummy_subject_ids)
        loss = out.sum()
        loss.backward()
        # Check that gradients exist for at least some parameters
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in encoder.parameters())
        assert has_grad

    def test_subject_embed_disable(self, dummy_eeg, dummy_subject_ids):
        encoder = EEGEncoder(embed_dim=128, n_temporal_transformer_layers=2)
        out_with = encoder(dummy_eeg, dummy_subject_ids, enable_subject_embed=True)
        out_without = encoder(dummy_eeg, dummy_subject_ids, enable_subject_embed=False)
        # Outputs should differ when subject embedding is toggled
        assert not torch.allclose(out_with, out_without, atol=1e-4)

    def test_without_frequency_branch(self, dummy_eeg, dummy_subject_ids):
        encoder = EEGEncoder(embed_dim=128, n_temporal_transformer_layers=2, use_frequency_branch=False)
        out = encoder(dummy_eeg, dummy_subject_ids)
        assert out.shape == (dummy_eeg.size(0), 768)


class TestProjectionHead:
    def test_output_normalized(self, dummy_embeddings):
        head = ProjectionHead(input_dim=768, hidden_dim=1024, output_dim=768)
        out = head(dummy_embeddings)
        norms = out.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_output_shape(self):
        head = ProjectionHead(input_dim=512, hidden_dim=1024, output_dim=768)
        x = torch.randn(4, 512)
        out = head(x)
        assert out.shape == (4, 768)


class TestSubjectEmbedding:
    def test_output_shape(self):
        emb = SubjectEmbedding(n_subjects=10, embed_dim=256)
        ids = torch.tensor([0, 3, 7, 9])
        out = emb(ids)
        assert out.shape == (4, 256)

    def test_different_subjects_different_embeddings(self):
        emb = SubjectEmbedding(n_subjects=10, embed_dim=256)
        e0 = emb(torch.tensor([0]))
        e1 = emb(torch.tensor([1]))
        assert not torch.allclose(e0, e1)
