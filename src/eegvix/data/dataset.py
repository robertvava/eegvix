"""THINGS-EEG2 dataset for paired EEG-image loading across multiple subjects."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ThingsEEG2Dataset(Dataset):
    """PyTorch Dataset for THINGS-EEG2 paired EEG-image data.

    Each sample returns:
        eeg: (17, 100) float tensor — EEG signal
        clip_embedding: (768,) float tensor — precomputed CLIP image embedding (if available)
        image: (3, 224, 224) float tensor — CLIP-preprocessed image (if clip_embeddings not precomputed)
        subject_id: int — subject index (0-9)
        image_id: int — unique image condition index
    """

    def __init__(
        self,
        data_dir: str | Path,
        subjects: list[int] | None = None,
        split: str = "train",
        indices: np.ndarray | None = None,
        average_repetitions: bool = True,
        clip_embeddings_path: str | Path | None = None,
        image_transform: transforms.Compose | None = None,
        eeg_transform: Callable | None = None,
        augment: bool = False,
    ):
        """
        Args:
            data_dir: Root directory containing preprocessed/ and images/ subdirectories.
            subjects: List of subject indices (0-9) to load. None = all 10.
            split: "train", "val", or "test".
            indices: For train/val, boolean mask or integer indices into the 16540 training conditions.
            average_repetitions: If True, average across EEG repetitions for cleaner signal.
            clip_embeddings_path: Path to precomputed CLIP embeddings (.pt file).
            image_transform: Torchvision transforms for images (used if no precomputed embeddings).
            eeg_transform: Callable transform for EEG data.
            augment: Whether to apply data augmentation to EEG.
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.average_repetitions = average_repetitions
        self.eeg_transform = eeg_transform
        self.augment = augment

        if subjects is None:
            subjects = list(range(10))
        self.subjects = subjects

        # Default CLIP-compatible image transform
        if image_transform is None:
            self.image_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],  # CLIP normalization
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ])
        else:
            self.image_transform = image_transform

        # Load precomputed CLIP embeddings if available
        self.clip_embeddings = None
        if clip_embeddings_path is not None and Path(clip_embeddings_path).exists():
            self.clip_embeddings = torch.load(clip_embeddings_path, weights_only=True)

        # Load EEG data for all subjects
        self._load_data(indices)

    def _load_data(self, indices: np.ndarray | None) -> None:
        """Load EEG data and build image path mappings."""
        self.eeg_data: list[torch.Tensor] = []
        self.subject_ids: list[int] = []
        self.image_ids: list[int] = []

        for subj_idx in self.subjects:
            subj_dir = self.data_dir / "preprocessed" / f"sub-{subj_idx + 1:02d}"

            if self.split == "test":
                raw = np.load(
                    subj_dir / "preprocessed_eeg_test.npy", allow_pickle=True
                ).item()
                eeg = raw["preprocessed_eeg_data"]  # (200, 80, 17, 100)

                if self.average_repetitions:
                    eeg = np.mean(eeg, axis=1)  # (200, 17, 100)
                    for img_idx in range(eeg.shape[0]):
                        self.eeg_data.append(torch.from_numpy(eeg[img_idx].astype(np.float32)))
                        self.subject_ids.append(subj_idx)
                        self.image_ids.append(img_idx)
                else:
                    for img_idx in range(eeg.shape[0]):
                        for rep in range(eeg.shape[1]):
                            self.eeg_data.append(torch.from_numpy(eeg[img_idx, rep].astype(np.float32)))
                            self.subject_ids.append(subj_idx)
                            self.image_ids.append(img_idx)
            else:
                raw = np.load(
                    subj_dir / "preprocessed_eeg_training.npy", allow_pickle=True
                ).item()
                eeg = raw["preprocessed_eeg_data"]  # (16540, 4, 17, 100)

                if self.average_repetitions:
                    eeg = np.mean(eeg, axis=1)  # (16540, 17, 100)

                # Apply train/val split
                if indices is not None:
                    if self.split == "val":
                        selected = indices
                    else:  # train
                        selected = ~indices if indices.dtype == bool else np.setdiff1d(np.arange(eeg.shape[0]), indices)
                else:
                    selected = np.arange(eeg.shape[0] if self.average_repetitions else eeg.shape[0])

                if self.average_repetitions:
                    for img_idx in (selected if selected.dtype != bool else np.where(selected)[0]):
                        self.eeg_data.append(torch.from_numpy(eeg[img_idx].astype(np.float32)))
                        self.subject_ids.append(subj_idx)
                        self.image_ids.append(int(img_idx))
                else:
                    for img_idx in (selected if selected.dtype != bool else np.where(selected)[0]):
                        for rep in range(eeg.shape[1]):
                            self.eeg_data.append(torch.from_numpy(eeg[img_idx, rep].astype(np.float32)))
                            self.subject_ids.append(subj_idx)
                            self.image_ids.append(int(img_idx))

        # Build image paths
        if self.split == "test":
            self.image_dir = self.data_dir / "images" / "test_images"
        else:
            self.image_dir = self.data_dir / "images" / "training_images"

        self._image_paths = self._build_image_paths()

    def _build_image_paths(self) -> list[Path]:
        """Collect and sort image paths from the image directory."""
        paths = sorted(self.image_dir.rglob("*.jpg"))
        if not paths:
            paths = sorted(self.image_dir.rglob("*.JPEG"))
        if not paths:
            paths = sorted(self.image_dir.rglob("*.png"))
        return paths

    def __len__(self) -> int:
        return len(self.eeg_data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int]:
        eeg = self.eeg_data[idx]
        subject_id = self.subject_ids[idx]
        image_id = self.image_ids[idx]

        if self.eeg_transform is not None:
            eeg = self.eeg_transform(eeg)

        sample = {
            "eeg": eeg,
            "subject_id": subject_id,
            "image_id": image_id,
        }

        # Prefer precomputed CLIP embeddings; fall back to loading image
        if self.clip_embeddings is not None:
            key = f"{self.split}_{image_id}"
            if key in self.clip_embeddings:
                sample["clip_embedding"] = self.clip_embeddings[key]
            elif image_id < len(self.clip_embeddings.get("embeddings", [])):
                sample["clip_embedding"] = self.clip_embeddings["embeddings"][image_id]
        else:
            if image_id < len(self._image_paths):
                img = Image.open(self._image_paths[image_id]).convert("RGB")
                sample["image"] = self.image_transform(img)

        return sample
