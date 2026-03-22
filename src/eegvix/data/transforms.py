"""EEG preprocessing transforms and image transforms for CLIP compatibility."""

import numpy as np
import torch


class BaselineCorrection:
    """Subtract mean of pre-stimulus interval from each channel.

    The dataset has 100 timepoints at 100Hz spanning -200ms to 800ms.
    Pre-stimulus interval: timepoints 0-19 (indices for -200ms to 0ms).
    """

    def __init__(self, n_pre_stimulus: int = 20):
        self.n_pre_stimulus = n_pre_stimulus

    def __call__(self, eeg: torch.Tensor) -> torch.Tensor:
        # eeg shape: (n_channels, n_timepoints)
        baseline = eeg[:, :self.n_pre_stimulus].mean(dim=1, keepdim=True)
        return eeg - baseline


class ChannelWiseZScore:
    """Z-score normalization per channel using precomputed statistics.

    Expects statistics computed across the training set for each channel.
    """

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        # mean, std shape: (n_channels, 1) or (n_channels,)
        self.mean = mean.view(-1, 1) if mean.dim() == 1 else mean
        self.std = std.view(-1, 1) if std.dim() == 1 else std

    def __call__(self, eeg: torch.Tensor) -> torch.Tensor:
        return (eeg - self.mean.to(eeg.device)) / (self.std.to(eeg.device) + 1e-8)


class RobustScaler:
    """Scale EEG using median and IQR per channel. More robust to artifacts than z-score."""

    def __init__(self, median: torch.Tensor, iqr: torch.Tensor):
        self.median = median.view(-1, 1)
        self.iqr = iqr.view(-1, 1)

    def __call__(self, eeg: torch.Tensor) -> torch.Tensor:
        return (eeg - self.median.to(eeg.device)) / (self.iqr.to(eeg.device) + 1e-8)


class TemporalJitter:
    """Data augmentation: randomly shift the EEG signal by a few timepoints."""

    def __init__(self, max_shift: int = 3):
        self.max_shift = max_shift

    def __call__(self, eeg: torch.Tensor) -> torch.Tensor:
        shift = torch.randint(-self.max_shift, self.max_shift + 1, (1,)).item()
        if shift == 0:
            return eeg
        if shift > 0:
            return torch.cat([eeg[:, shift:], torch.zeros_like(eeg[:, :shift])], dim=1)
        return torch.cat([torch.zeros_like(eeg[:, :abs(shift)]), eeg[:, :shift]], dim=1)


class GaussianNoise:
    """Data augmentation: add Gaussian noise to EEG."""

    def __init__(self, std: float = 0.01):
        self.std = std

    def __call__(self, eeg: torch.Tensor) -> torch.Tensor:
        return eeg + torch.randn_like(eeg) * self.std


class ChannelDropout:
    """Data augmentation: randomly zero out entire channels."""

    def __init__(self, p: float = 0.1):
        self.p = p

    def __call__(self, eeg: torch.Tensor) -> torch.Tensor:
        mask = torch.bernoulli(torch.full((eeg.size(0), 1), 1.0 - self.p))
        return eeg * mask


class BandpowerFeatures:
    """Extract power in standard EEG frequency bands via FFT.

    Bands: delta (1-4Hz), theta (4-8Hz), alpha (8-13Hz), beta (13-30Hz), gamma (30-50Hz).
    At 100Hz sampling rate with 100 timepoints, frequency resolution is 1Hz.
    """

    BANDS: dict[str, tuple[float, float]] = {
        "delta": (1.0, 4.0),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 13.0),
        "beta": (13.0, 30.0),
        "gamma": (30.0, 50.0),
    }

    def __init__(self, sampling_rate: int = 100, n_timepoints: int = 100):
        self.sampling_rate = sampling_rate
        self.n_timepoints = n_timepoints
        self.freqs = np.fft.rfftfreq(n_timepoints, d=1.0 / sampling_rate)

    def __call__(self, eeg: torch.Tensor) -> torch.Tensor:
        """Extract bandpower features.

        Args:
            eeg: (n_channels, n_timepoints)

        Returns:
            bandpowers: (n_channels, n_bands) where n_bands=5
        """
        fft_vals = torch.fft.rfft(eeg, dim=-1)
        power = (fft_vals.real ** 2 + fft_vals.imag ** 2)

        bandpowers = []
        for low, high in self.BANDS.values():
            mask = torch.tensor(
                (self.freqs >= low) & (self.freqs < high), dtype=torch.float32
            )
            band_power = (power * mask.to(eeg.device)).mean(dim=-1)
            bandpowers.append(band_power)

        return torch.stack(bandpowers, dim=-1)  # (n_channels, 5)


class ComposeEEGTransforms:
    """Compose multiple EEG transforms sequentially."""

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, eeg: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            eeg = t(eeg)
        return eeg
