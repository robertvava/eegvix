"""Generate images from EEG signals using the full pipeline.

Usage:
    python scripts/generate.py --checkpoint path/to/contrastive_checkpoint.ckpt --data_dir eeg_dataset
"""

import argparse
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

from eegvix.generation.pipeline import EEGToImagePipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Contrastive training checkpoint")
    parser.add_argument("--data_dir", type=str, default="eeg_dataset")
    parser.add_argument("--output_dir", type=str, default="generated_images")
    parser.add_argument("--subject", type=int, default=0)
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--n_images", type=int, default=5, help="Images per EEG condition")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--sd_model", type=str, default="stabilityai/stable-diffusion-2-1")
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = EEGToImagePipeline.from_pretrained(
        contrastive_checkpoint=args.checkpoint,
        sd_model=args.sd_model,
        lora_path=args.lora_path,
        device=args.device,
    )

    # Load test EEG data
    data_dir = Path(args.data_dir)
    subj_dir = data_dir / "preprocessed" / f"sub-{args.subject + 1:02d}"

    if args.split == "test":
        raw = np.load(subj_dir / "preprocessed_eeg_test.npy", allow_pickle=True).item()
    else:
        raw = np.load(subj_dir / "preprocessed_eeg_training.npy", allow_pickle=True).item()

    eeg_data = raw["preprocessed_eeg_data"]
    # Average across repetitions for cleaner signal
    eeg_data = np.mean(eeg_data, axis=1)

    print(f"Generating images for {eeg_data.shape[0]} conditions...")

    for i in tqdm(range(eeg_data.shape[0])):
        eeg_tensor = torch.from_numpy(eeg_data[i].astype(np.float32))
        images = pipeline.generate(
            eeg_tensor,
            subject_id=args.subject,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            num_images=args.n_images,
        )

        for j, img in enumerate(images):
            img.save(output_dir / f"condition_{i:05d}_sample_{j}.png")


if __name__ == "__main__":
    main()
