"""InfoNCE / CLIP-style symmetric contrastive loss.

The loss aligns EEG embeddings with their corresponding CLIP image embeddings
in a shared space. Within each batch, matching EEG-image pairs are positives;
all other combinations are negatives.

L = 0.5 * (CE(eeg_to_img_logits, labels) + CE(img_to_eeg_logits, labels))

where logits[i,j] = (eeg_embed[i] . img_embed[j]) / temperature
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """Symmetric InfoNCE contrastive loss with learnable temperature.

    Equivalent to the CLIP loss: matches EEG embeddings to image embeddings
    bidirectionally within a batch.
    """

    def __init__(self, init_temperature: float = 0.07, learnable: bool = True):
        super().__init__()
        # log_temperature is learned; temperature = exp(log_temperature)
        log_temp = torch.tensor([-torch.tensor(init_temperature).log()])
        if learnable:
            self.log_temperature = nn.Parameter(log_temp)
        else:
            self.register_buffer("log_temperature", log_temp)

    @property
    def temperature(self) -> torch.Tensor:
        # Clamp to avoid numerical instability
        return self.log_temperature.exp().clamp(min=1e-4, max=100.0)

    def forward(
        self,
        eeg_embeds: torch.Tensor,
        image_embeds: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute symmetric contrastive loss.

        Args:
            eeg_embeds: (batch, embed_dim) L2-normalized EEG embeddings
            image_embeds: (batch, embed_dim) L2-normalized image embeddings

        Returns:
            dict with keys: "loss", "eeg_to_img_acc", "img_to_eeg_acc", "temperature"
        """
        # Cosine similarity scaled by temperature
        logits = (eeg_embeds @ image_embeds.T) / self.temperature  # (batch, batch)
        labels = torch.arange(logits.size(0), device=logits.device)

        # Symmetric cross-entropy
        loss_eeg_to_img = F.cross_entropy(logits, labels)
        loss_img_to_eeg = F.cross_entropy(logits.T, labels)
        loss = (loss_eeg_to_img + loss_img_to_eeg) / 2

        # Accuracy metrics (for logging)
        with torch.no_grad():
            eeg_to_img_acc = (logits.argmax(dim=1) == labels).float().mean()
            img_to_eeg_acc = (logits.T.argmax(dim=1) == labels).float().mean()

        return {
            "loss": loss,
            "eeg_to_img_acc": eeg_to_img_acc,
            "img_to_eeg_acc": img_to_eeg_acc,
            "temperature": self.temperature.detach(),
        }
