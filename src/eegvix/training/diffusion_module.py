"""PyTorch Lightning module for optional LoRA fine-tuning of Stable Diffusion.

This module fine-tunes the cross-attention layers of the UNet using LoRA,
conditioned on EEG-derived CLIP embeddings paired with their ground truth images.
"""

import torch
import lightning as L
from torch.optim import AdamW
from diffusers import DDPMScheduler

from eegvix.models.eeg_encoder import EEGEncoder
from eegvix.models.projection_head import ProjectionHead


class DiffusionFineTuneModule(L.LightningModule):
    """Fine-tunes Stable Diffusion's UNet cross-attention via LoRA.

    Loads a pretrained EEG encoder (from contrastive training) and uses it
    to produce conditioning embeddings for the diffusion model.
    """

    def __init__(
        self,
        eeg_encoder_checkpoint: str,
        sd_model: str = "stabilityai/stable-diffusion-2-1",
        lora_rank: int = 16,
        lora_alpha: int = 32,
        learning_rate: float = 1e-5,
        weight_decay: float = 0.01,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Load frozen EEG encoder from contrastive training
        self.eeg_encoder = EEGEncoder()
        self.projection_head = ProjectionHead()
        checkpoint = torch.load(eeg_encoder_checkpoint, map_location="cpu", weights_only=False)
        if "state_dict" in checkpoint:
            state = checkpoint["state_dict"]
            encoder_state = {k.replace("eeg_encoder.", ""): v for k, v in state.items() if k.startswith("eeg_encoder.")}
            proj_state = {k.replace("projection_head.", ""): v for k, v in state.items() if k.startswith("projection_head.")}
            self.eeg_encoder.load_state_dict(encoder_state)
            self.projection_head.load_state_dict(proj_state)

        for param in self.eeg_encoder.parameters():
            param.requires_grad = False
        for param in self.projection_head.parameters():
            param.requires_grad = False
        self.eeg_encoder.eval()
        self.projection_head.eval()

        # Noise scheduler for training
        self.noise_scheduler = DDPMScheduler.from_pretrained(sd_model, subfolder="scheduler")

        # The actual UNet + LoRA setup happens in setup() to defer heavy model loading
        self.sd_model = sd_model
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self._unet = None
        self._vae = None

    def setup(self, stage: str | None = None) -> None:
        if self._unet is not None:
            return

        from diffusers import AutoencoderKL, UNet2DConditionModel
        from peft import LoraConfig, get_peft_model

        self._vae = AutoencoderKL.from_pretrained(self.sd_model, subfolder="vae")
        self._vae.requires_grad_(False)
        self._vae.eval()

        self._unet = UNet2DConditionModel.from_pretrained(self.sd_model, subfolder="unet")
        lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            target_modules=["to_k", "to_v", "to_q", "to_out.0"],
            lora_dropout=0.05,
        )
        self._unet = get_peft_model(self._unet, lora_config)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        images = batch["image"]  # (batch, 3, 512, 512)
        eeg = batch["eeg"]
        subject_ids = batch["subject_id"]

        # Get EEG-derived CLIP embeddings (frozen)
        with torch.no_grad():
            eeg_features = self.eeg_encoder(eeg, subject_ids)
            eeg_embeds = self.projection_head(eeg_features)

        # Encode images to latent space
        with torch.no_grad():
            latents = self._vae.encode(images).latent_dist.sample()
            latents = latents * self._vae.config.scaling_factor

        # Add noise
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (latents.size(0),), device=self.device)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # Predict noise conditioned on EEG embeddings
        # Reshape embeddings to match expected cross-attention input: (batch, seq_len, dim)
        encoder_hidden_states = eeg_embeds.unsqueeze(1).expand(-1, 77, -1)

        noise_pred = self._unet(noisy_latents, timesteps, encoder_hidden_states).sample
        loss = torch.nn.functional.mse_loss(noise_pred, noise)

        self.log("train/diffusion_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # Only optimize LoRA parameters
        trainable_params = [p for p in self._unet.parameters() if p.requires_grad]
        optimizer = AdamW(trainable_params, lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer
