"""Electrode positions for the 17 occipital/parietal channels used in THINGS-EEG2.

2D positions are approximate projections onto a unit circle (top-down view of scalp).
Coordinates follow the standard 10-10 system layout used in the dataset.
"""

import torch

# Channel names in the order they appear in the preprocessed data
CHANNEL_NAMES: list[str] = [
    "O1", "Oz", "O2",
    "PO7", "PO3", "POz", "PO4", "PO8",
    "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
]

# Approximate 2D scalp positions (x, y) in normalized coordinates [-1, 1].
# x: left(-) to right(+), y: posterior(-) to anterior(+)
# These follow the standard 10-10 montage layout.
CHANNEL_POSITIONS_2D: dict[str, tuple[float, float]] = {
    "O1":  (-0.31, -0.95),
    "Oz":  ( 0.00, -1.00),
    "O2":  ( 0.31, -0.95),
    "PO7": (-0.59, -0.81),
    "PO3": (-0.31, -0.81),
    "POz": ( 0.00, -0.81),
    "PO4": ( 0.31, -0.81),
    "PO8": ( 0.59, -0.81),
    "P7":  (-0.81, -0.59),
    "P5":  (-0.59, -0.59),
    "P3":  (-0.39, -0.59),
    "P1":  (-0.19, -0.59),
    "Pz":  ( 0.00, -0.59),
    "P2":  ( 0.19, -0.59),
    "P4":  ( 0.39, -0.59),
    "P6":  ( 0.59, -0.59),
    "P8":  ( 0.81, -0.59),
}

N_CHANNELS = len(CHANNEL_NAMES)


def get_channel_positions_tensor() -> torch.Tensor:
    """Return channel positions as a (17, 2) float tensor, ordered by CHANNEL_NAMES."""
    positions = [CHANNEL_POSITIONS_2D[ch] for ch in CHANNEL_NAMES]
    return torch.tensor(positions, dtype=torch.float32)
