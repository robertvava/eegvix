"""Frozen CLIP ViT-L/14 image encoder wrapper.

Provides a simple interface to extract CLIP image embeddings,
either online during training or as a precomputation step.
"""

from pathlib import Path

import torch
import torch.nn as nn
import open_clip


class CLIPImageEncoder(nn.Module):
    """Wraps OpenCLIP's image encoder with all parameters frozen.

    The forward pass returns L2-normalized image embeddings
    in the same space that the EEG encoder is trained to match.
    """

    def __init__(self, model_name: str = "ViT-L-14", pretrained: str = "openai"):
        super().__init__()
        model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.visual = model.visual
        self.embed_dim = model.visual.output_dim

        # Freeze all parameters
        for param in self.visual.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Extract CLIP image embeddings.

        Args:
            images: (batch, 3, 224, 224) preprocessed images

        Returns:
            (batch, embed_dim) L2-normalized image embeddings
        """
        features = self.visual(images)
        return nn.functional.normalize(features, dim=-1)

    @torch.no_grad()
    def precompute_embeddings(
        self,
        image_paths: list[Path],
        batch_size: int = 64,
        device: str = "cuda",
    ) -> torch.Tensor:
        """Precompute CLIP embeddings for a list of image files.

        Args:
            image_paths: List of paths to images.
            batch_size: Processing batch size.
            device: Device to run inference on.

        Returns:
            (n_images, embed_dim) tensor of embeddings.
        """
        from PIL import Image
        from tqdm import tqdm

        self.to(device)
        self.eval()

        all_embeddings = []
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Precomputing CLIP embeddings"):
            batch_paths = image_paths[i : i + batch_size]
            batch_images = []
            for p in batch_paths:
                img = Image.open(p).convert("RGB")
                batch_images.append(self.preprocess(img))

            batch_tensor = torch.stack(batch_images).to(device)
            embeddings = self.forward(batch_tensor)
            all_embeddings.append(embeddings.cpu())

        return torch.cat(all_embeddings, dim=0)
