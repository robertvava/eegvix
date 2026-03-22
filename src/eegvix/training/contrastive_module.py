"""PyTorch Lightning module for contrastive EEG-CLIP alignment training."""

import torch
import lightning as L
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR

from eegvix.models.eeg_encoder import EEGEncoder
from eegvix.models.clip_wrapper import CLIPImageEncoder
from eegvix.models.projection_head import ProjectionHead
from eegvix.losses.contrastive import InfoNCELoss


class ContrastiveAlignmentModule(L.LightningModule):
    """Trains the EEG encoder to produce CLIP-aligned embeddings via InfoNCE loss.

    The CLIP image encoder is frozen. Only the EEG encoder and projection head
    are trained. Optionally uses precomputed CLIP embeddings to save memory.
    """

    def __init__(
        self,
        # EEG encoder config
        n_channels: int = 17,
        n_timepoints: int = 100,
        embed_dim: int = 512,
        num_temporal_conv_layers: int = 3,
        temporal_kernel_sizes: list[int] | None = None,
        n_spatial_heads: int = 4,
        n_temporal_transformer_layers: int = 4,
        n_temporal_heads: int = 8,
        dropout: float = 0.1,
        use_frequency_branch: bool = True,
        output_dim: int = 768,
        n_subjects: int = 10,
        # Projection head config
        proj_hidden_dim: int = 2048,
        # Loss config
        init_temperature: float = 0.07,
        learnable_temperature: bool = True,
        # Training config
        learning_rate: float = 3e-4,
        weight_decay: float = 0.01,
        warmup_epochs: int = 10,
        max_epochs: int = 200,
        subject_embedding_warmup_epochs: int = 5,
        # CLIP config
        clip_model_name: str = "ViT-L-14",
        clip_pretrained: str = "openai",
        use_precomputed_clip: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        # EEG encoder (trainable)
        self.eeg_encoder = EEGEncoder(
            n_channels=n_channels,
            n_timepoints=n_timepoints,
            embed_dim=embed_dim,
            num_temporal_conv_layers=num_temporal_conv_layers,
            temporal_kernel_sizes=temporal_kernel_sizes,
            n_spatial_heads=n_spatial_heads,
            n_temporal_transformer_layers=n_temporal_transformer_layers,
            n_temporal_heads=n_temporal_heads,
            dropout=dropout,
            use_frequency_branch=use_frequency_branch,
            output_dim=output_dim,
            n_subjects=n_subjects,
        )

        # Projection head (trainable)
        self.projection_head = ProjectionHead(
            input_dim=output_dim,
            hidden_dim=proj_hidden_dim,
            output_dim=output_dim,
        )

        # CLIP image encoder (frozen, only needed if not using precomputed)
        self.use_precomputed_clip = use_precomputed_clip
        if not use_precomputed_clip:
            self.clip_encoder = CLIPImageEncoder(
                model_name=clip_model_name, pretrained=clip_pretrained
            )

        # Loss
        self.criterion = InfoNCELoss(
            init_temperature=init_temperature,
            learnable=learnable_temperature,
        )

        # Config
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.subject_embedding_warmup_epochs = subject_embedding_warmup_epochs

    def forward(self, eeg: torch.Tensor, subject_ids: torch.Tensor) -> torch.Tensor:
        """Encode EEG and project to CLIP space.

        Returns L2-normalized embeddings suitable for contrastive loss or retrieval.
        """
        enable_subject = self.current_epoch >= self.subject_embedding_warmup_epochs
        eeg_features = self.eeg_encoder(eeg, subject_ids, enable_subject_embed=enable_subject)
        return self.projection_head(eeg_features)

    def _shared_step(self, batch: dict, stage: str) -> torch.Tensor:
        eeg = batch["eeg"]
        subject_ids = batch["subject_id"]

        # Get EEG embeddings
        eeg_embeds = self.forward(eeg, subject_ids)

        # Get image embeddings
        if self.use_precomputed_clip and "clip_embedding" in batch:
            image_embeds = batch["clip_embedding"]
            image_embeds = torch.nn.functional.normalize(image_embeds, dim=-1)
        else:
            image_embeds = self.clip_encoder(batch["image"])

        # Contrastive loss
        loss_dict = self.criterion(eeg_embeds, image_embeds)

        # Log metrics
        self.log(f"{stage}/loss", loss_dict["loss"], prog_bar=True, sync_dist=True)
        self.log(f"{stage}/eeg_to_img_acc", loss_dict["eeg_to_img_acc"], sync_dist=True)
        self.log(f"{stage}/img_to_eeg_acc", loss_dict["img_to_eeg_acc"], sync_dist=True)
        self.log(f"{stage}/temperature", loss_dict["temperature"], sync_dist=True)

        return loss_dict["loss"]

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def test_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        # Separate parameter groups: encoder, projection head, temperature
        param_groups = [
            {"params": self.eeg_encoder.parameters(), "lr": self.learning_rate},
            {"params": self.projection_head.parameters(), "lr": self.learning_rate},
            {"params": self.criterion.parameters(), "lr": self.learning_rate * 10},
        ]

        optimizer = AdamW(param_groups, weight_decay=self.weight_decay)

        # Linear warmup + cosine decay
        warmup_scheduler = LinearLR(
            optimizer, start_factor=0.01, total_iters=self.warmup_epochs
        )
        cosine_scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=self.max_epochs - self.warmup_epochs, T_mult=1
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.warmup_epochs],
        )

        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
