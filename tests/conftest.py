"""Shared test fixtures for EEGVIX tests."""

import pytest
import torch


@pytest.fixture
def batch_size():
    return 4


@pytest.fixture
def n_channels():
    return 17


@pytest.fixture
def n_timepoints():
    return 100


@pytest.fixture
def embed_dim():
    return 512


@pytest.fixture
def output_dim():
    return 768


@pytest.fixture
def dummy_eeg(batch_size, n_channels, n_timepoints):
    """Random EEG tensor simulating a batch."""
    return torch.randn(batch_size, n_channels, n_timepoints)


@pytest.fixture
def dummy_subject_ids(batch_size):
    """Random subject IDs in [0, 9]."""
    return torch.randint(0, 10, (batch_size,))


@pytest.fixture
def dummy_images(batch_size):
    """Random image tensor (CLIP-preprocessed size)."""
    return torch.randn(batch_size, 3, 224, 224)


@pytest.fixture
def dummy_embeddings(batch_size, output_dim):
    """Random L2-normalized embeddings."""
    x = torch.randn(batch_size, output_dim)
    return torch.nn.functional.normalize(x, dim=-1)
