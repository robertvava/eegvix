"""Precompute CLIP image embeddings for all training and test images.

Saves embeddings as .pt files to avoid running CLIP during contrastive training.

Usage:
    python scripts/precompute_clip.py --data_dir eeg_dataset --output_dir eeg_dataset/clip_embeddings
"""

import argparse
from pathlib import Path

import torch

from eegvix.models.clip_wrapper import CLIPImageEncoder


def main():
    parser = argparse.ArgumentParser(description="Precompute CLIP image embeddings")
    parser.add_argument("--data_dir", type=str, default="eeg_dataset")
    parser.add_argument("--output_dir", type=str, default="eeg_dataset/clip_embeddings")
    parser.add_argument("--model_name", type=str, default="ViT-L-14")
    parser.add_argument("--pretrained", type=str, default="openai")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    clip = CLIPImageEncoder(model_name=args.model_name, pretrained=args.pretrained)

    for split, subdir in [("train", "training_images"), ("test", "test_images")]:
        image_dir = data_dir / "images" / subdir
        image_paths = sorted(image_dir.rglob("*.jpg"))
        if not image_paths:
            image_paths = sorted(image_dir.rglob("*.JPEG"))
        if not image_paths:
            image_paths = sorted(image_dir.rglob("*.png"))

        if not image_paths:
            print(f"No images found in {image_dir}")
            continue

        print(f"Computing {split} embeddings for {len(image_paths)} images...")
        embeddings = clip.precompute_embeddings(
            image_paths, batch_size=args.batch_size, device=args.device
        )

        output_path = output_dir / f"{split}_clip_embeddings.pt"
        torch.save({"embeddings": embeddings, "paths": [str(p) for p in image_paths]}, output_path)
        print(f"Saved {split} embeddings to {output_path} — shape: {embeddings.shape}")


if __name__ == "__main__":
    main()
