"""End-to-end EEG-to-image generation pipeline.

Usage:
    pipeline = EEGToImagePipeline.from_pretrained("path/to/checkpoint")
    images = pipeline.generate(eeg_tensor, subject_id=0)
"""

from pathlib import Path

import torch
from PIL import Image

from eegvix.models.eeg_encoder import EEGEncoder
from eegvix.models.projection_head import ProjectionHead
from eegvix.models.diffusion_wrapper import DiffusionWrapper


class EEGToImagePipeline:
    """Complete pipeline: raw EEG → CLIP embedding → Stable Diffusion → image."""

    def __init__(
        self,
        eeg_encoder: EEGEncoder,
        projection_head: ProjectionHead,
        diffusion: DiffusionWrapper,
        device: str = "cuda",
    ):
        self.device = device
        self.eeg_encoder = eeg_encoder.to(device).eval()
        self.projection_head = projection_head.to(device).eval()
        self.diffusion = diffusion

    @classmethod
    def from_pretrained(
        cls,
        contrastive_checkpoint: str | Path,
        sd_model: str = "stabilityai/stable-diffusion-2-1",
        use_ip_adapter: bool = True,
        lora_path: str | Path | None = None,
        device: str = "cuda",
    ) -> "EEGToImagePipeline":
        """Load a pretrained pipeline from a contrastive training checkpoint.

        Args:
            contrastive_checkpoint: Path to the Lightning checkpoint from contrastive training.
            sd_model: Stable Diffusion model identifier.
            use_ip_adapter: Whether to use IP-Adapter for conditioning.
            lora_path: Optional path to LoRA weights for the diffusion model.
            device: Device to load models on.
        """
        checkpoint = torch.load(contrastive_checkpoint, map_location="cpu", weights_only=False)
        state = checkpoint.get("state_dict", checkpoint)

        # Extract EEG encoder state
        encoder_state = {
            k.replace("eeg_encoder.", ""): v
            for k, v in state.items()
            if k.startswith("eeg_encoder.")
        }
        proj_state = {
            k.replace("projection_head.", ""): v
            for k, v in state.items()
            if k.startswith("projection_head.")
        }

        eeg_encoder = EEGEncoder()
        eeg_encoder.load_state_dict(encoder_state)

        projection_head = ProjectionHead()
        projection_head.load_state_dict(proj_state)

        diffusion = DiffusionWrapper(
            sd_model=sd_model,
            use_ip_adapter=use_ip_adapter,
            use_lora=lora_path is not None,
            device=device,
        )
        if lora_path is not None:
            diffusion.load_lora(lora_path)

        return cls(eeg_encoder, projection_head, diffusion, device)

    @torch.no_grad()
    def encode_eeg(
        self,
        eeg: torch.Tensor,
        subject_id: int | torch.Tensor = 0,
    ) -> torch.Tensor:
        """Encode raw EEG to a CLIP-space embedding.

        Args:
            eeg: (batch, 17, 100) or (17, 100) raw EEG signal
            subject_id: Subject index or tensor of indices

        Returns:
            (batch, 768) L2-normalized CLIP-space embeddings
        """
        if eeg.dim() == 2:
            eeg = eeg.unsqueeze(0)
        eeg = eeg.to(self.device)

        if isinstance(subject_id, int):
            subject_ids = torch.full((eeg.size(0),), subject_id, device=self.device, dtype=torch.long)
        else:
            subject_ids = subject_id.to(self.device)

        features = self.eeg_encoder(eeg, subject_ids)
        return self.projection_head(features)

    @torch.no_grad()
    def generate(
        self,
        eeg: torch.Tensor,
        subject_id: int | torch.Tensor = 0,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        num_images: int = 1,
    ) -> list[Image.Image]:
        """Generate images from raw EEG signals.

        Args:
            eeg: (batch, 17, 100) or (17, 100) raw EEG signal
            subject_id: Subject index
            num_inference_steps: Diffusion denoising steps
            guidance_scale: Classifier-free guidance scale
            num_images: Number of images to generate per EEG sample

        Returns:
            List of PIL images
        """
        clip_embedding = self.encode_eeg(eeg, subject_id)

        return self.diffusion.generate(
            clip_embedding=clip_embedding,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images,
        )
