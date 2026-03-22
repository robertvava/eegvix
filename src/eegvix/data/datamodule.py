"""PyTorch Lightning DataModule for THINGS-EEG2."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import lightning as L
from sklearn.utils import resample
from torch.utils.data import DataLoader

from eegvix.data.dataset import ThingsEEG2Dataset


class ThingsEEG2DataModule(L.LightningDataModule):
    """Lightning DataModule that handles train/val/test splits for THINGS-EEG2.

    Uses concept-based stratified splitting: entire concepts (10 images each)
    are assigned to validation, preserving semantic structure.
    """

    def __init__(
        self,
        data_dir: str | Path,
        batch_size: int = 256,
        num_workers: int = 4,
        subjects: list[int] | None = None,
        average_repetitions: bool = True,
        val_n_concepts: int = 150,
        val_random_state: int = 42,
        clip_embeddings_path: str | Path | None = None,
        eeg_transform: Callable | None = None,
        augment_train: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["eeg_transform"])
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.subjects = subjects
        self.average_repetitions = average_repetitions
        self.val_n_concepts = val_n_concepts
        self.val_random_state = val_random_state
        self.clip_embeddings_path = clip_embeddings_path
        self.eeg_transform = eeg_transform
        self.augment_train = augment_train

    def _get_val_indices(self) -> np.ndarray:
        """Create concept-based validation split (same logic as original codebase)."""
        n_concepts = 1654
        images_per_concept = 10
        n_conditions = n_concepts * images_per_concept  # 16540

        val_concepts = np.sort(
            resample(
                np.arange(n_concepts),
                replace=False,
                n_samples=self.val_n_concepts,
                random_state=self.val_random_state,
            )
        )

        idx_val = np.zeros(n_conditions, dtype=bool)
        for c in val_concepts:
            idx_val[c * images_per_concept : (c + 1) * images_per_concept] = True

        return idx_val

    def setup(self, stage: str | None = None) -> None:
        val_indices = self._get_val_indices()

        if stage in ("fit", None):
            self.train_dataset = ThingsEEG2Dataset(
                data_dir=self.data_dir,
                subjects=self.subjects,
                split="train",
                indices=val_indices,
                average_repetitions=self.average_repetitions,
                clip_embeddings_path=self.clip_embeddings_path,
                eeg_transform=self.eeg_transform,
                augment=self.augment_train,
            )
            self.val_dataset = ThingsEEG2Dataset(
                data_dir=self.data_dir,
                subjects=self.subjects,
                split="val",
                indices=val_indices,
                average_repetitions=self.average_repetitions,
                clip_embeddings_path=self.clip_embeddings_path,
                eeg_transform=self.eeg_transform,
                augment=False,
            )

        if stage in ("test", None):
            self.test_dataset = ThingsEEG2Dataset(
                data_dir=self.data_dir,
                subjects=self.subjects,
                split="test",
                average_repetitions=self.average_repetitions,
                clip_embeddings_path=self.clip_embeddings_path,
                eeg_transform=self.eeg_transform,
                augment=False,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
