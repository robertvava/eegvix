"""Stable Diffusion wrapper with IP-Adapter and optional LoRA for EEG-conditioned generation.

IP-Adapter injects image embeddings (or in our case, EEG-derived CLIP embeddings) into
the cross-attention layers of the UNet, enabling image-conditioned generation without
modifying the diffusion model's architecture.
"""

from pathlib import Path

import torch
import torch.nn as nn
from diffusers import StableDiffusionPipeline, DDIMScheduler


class DiffusionWrapper(nn.Module):
    """Wraps Stable Diffusion for CLIP-embedding-conditioned image generation.

    Supports:
        - IP-Adapter conditioning (primary approach)
        - Optional LoRA fine-tuning of cross-attention layers
    """

    def __init__(
        self,
        sd_model: str = "stabilityai/stable-diffusion-2-1",
        use_ip_adapter: bool = True,
        use_lora: bool = False,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        device: str = "cuda",
    ):
        super().__init__()
        self.device = device
        self.use_ip_adapter = use_ip_adapter
        self.use_lora = use_lora

        # Load Stable Diffusion pipeline
        self.pipe = StableDiffusionPipeline.from_pretrained(
            sd_model,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe = self.pipe.to(device)

        # Load IP-Adapter if requested
        if use_ip_adapter:
            self.pipe.load_ip_adapter(
                "h94/IP-Adapter",
                subfolder="models",
                weight_name="ip-adapter_sd15.bin",
            )
            self.pipe.set_ip_adapter_scale(0.8)

        # Apply LoRA if requested
        if use_lora:
            from peft import LoraConfig, get_peft_model

            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                target_modules=["to_k", "to_v", "to_q", "to_out.0"],
                lora_dropout=0.05,
            )
            self.pipe.unet = get_peft_model(self.pipe.unet, lora_config)

    @torch.no_grad()
    def generate(
        self,
        clip_embedding: torch.Tensor,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        num_images_per_prompt: int = 1,
        height: int = 512,
        width: int = 512,
    ) -> list:
        """Generate images conditioned on CLIP embeddings (from EEG encoder).

        Args:
            clip_embedding: (batch, 768) CLIP-space embeddings from the EEG encoder.
            num_inference_steps: Number of diffusion denoising steps.
            guidance_scale: Classifier-free guidance scale.
            num_images_per_prompt: Number of images to generate per embedding.
            height: Output image height.
            width: Output image width.

        Returns:
            List of PIL images.
        """
        if self.use_ip_adapter:
            # IP-Adapter expects image embeddings
            # We pass EEG-derived CLIP embeddings as if they were image embeddings
            output = self.pipe(
                prompt="",
                ip_adapter_image_embeds=[clip_embedding.to(self.device, dtype=torch.float16)],
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                num_images_per_prompt=num_images_per_prompt,
                height=height,
                width=width,
            )
        else:
            # Direct text-encoder replacement: project CLIP embedding into text encoder space
            # This is a simpler but less effective approach
            prompt_embeds = clip_embedding.unsqueeze(1).expand(-1, 77, -1)
            prompt_embeds = prompt_embeds.to(self.device, dtype=torch.float16)

            output = self.pipe(
                prompt_embeds=prompt_embeds,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                num_images_per_prompt=num_images_per_prompt,
                height=height,
                width=width,
            )

        return output.images

    def save_lora(self, path: str | Path) -> None:
        if self.use_lora:
            self.pipe.unet.save_pretrained(path)

    def load_lora(self, path: str | Path) -> None:
        if self.use_lora:
            from peft import PeftModel
            self.pipe.unet = PeftModel.from_pretrained(self.pipe.unet, path)
