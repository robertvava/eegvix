"""Run the full evaluation suite on a trained contrastive model.

Usage:
    python scripts/evaluate.py --checkpoint path/to/checkpoint.ckpt --data_dir eeg_dataset
"""

import argparse
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

from eegvix.models.eeg_encoder import EEGEncoder
from eegvix.models.projection_head import ProjectionHead
from eegvix.models.clip_wrapper import CLIPImageEncoder
from eegvix.evaluation.retrieval import top_k_accuracy, zero_shot_identification
from eegvix.evaluation.rsa import rsa_correlation


def load_model(checkpoint_path: str, device: str = "cuda"):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)

    encoder_state = {k.replace("eeg_encoder.", ""): v for k, v in state.items() if k.startswith("eeg_encoder.")}
    proj_state = {k.replace("projection_head.", ""): v for k, v in state.items() if k.startswith("projection_head.")}

    eeg_encoder = EEGEncoder()
    eeg_encoder.load_state_dict(encoder_state)
    eeg_encoder.to(device).eval()

    projection_head = ProjectionHead()
    projection_head.load_state_dict(proj_state)
    projection_head.to(device).eval()

    return eeg_encoder, projection_head


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="eeg_dataset")
    parser.add_argument("--clip_embeddings", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--subjects", type=int, nargs="+", default=None)
    args = parser.parse_args()

    device = args.device
    data_dir = Path(args.data_dir)
    subjects = args.subjects or list(range(10))

    eeg_encoder, projection_head = load_model(args.checkpoint, device)

    # Load CLIP image embeddings
    if args.clip_embeddings:
        clip_data = torch.load(args.clip_embeddings, weights_only=True)
        test_image_embeds = clip_data["embeddings"]
    else:
        print("Computing CLIP embeddings for test images...")
        clip = CLIPImageEncoder()
        image_dir = data_dir / "images" / "test_images"
        image_paths = sorted(image_dir.rglob("*.jpg"))
        if not image_paths:
            image_paths = sorted(image_dir.rglob("*.JPEG"))
        test_image_embeds = clip.precompute_embeddings(image_paths, device=device)

    print(f"\nEvaluating on {len(subjects)} subjects...")
    print("=" * 60)

    all_results = {}

    for subj_idx in subjects:
        subj_dir = data_dir / "preprocessed" / f"sub-{subj_idx + 1:02d}"
        raw = np.load(subj_dir / "preprocessed_eeg_test.npy", allow_pickle=True).item()
        eeg_data = raw["preprocessed_eeg_data"]  # (200, 80, 17, 100)
        eeg_averaged = np.mean(eeg_data, axis=1)  # (200, 17, 100)

        # Encode EEG
        eeg_tensor = torch.from_numpy(eeg_averaged.astype(np.float32)).to(device)
        subject_ids = torch.full((eeg_tensor.size(0),), subj_idx, device=device, dtype=torch.long)

        with torch.no_grad():
            features = eeg_encoder(eeg_tensor, subject_ids)
            eeg_embeds = projection_head(features)

        img_embeds = test_image_embeds.to(device)

        # Top-k retrieval
        retrieval = top_k_accuracy(eeg_embeds, img_embeds)

        # Zero-shot identification (200-way)
        zs = zero_shot_identification(eeg_embeds, img_embeds)

        # RSA
        rsa = rsa_correlation(eeg_embeds, img_embeds)

        results = {**retrieval, "zero_shot_acc": zs["accuracy"], **rsa}
        all_results[f"sub-{subj_idx + 1:02d}"] = results

        print(f"\nSubject {subj_idx + 1:02d}:")
        for k, v in results.items():
            print(f"  {k}: {v:.4f}")

    # Average across subjects
    print("\n" + "=" * 60)
    print("Average across subjects:")
    avg_keys = list(next(iter(all_results.values())).keys())
    for k in avg_keys:
        values = [r[k] for r in all_results.values()]
        print(f"  {k}: {np.mean(values):.4f} ± {np.std(values):.4f}")


if __name__ == "__main__":
    main()
