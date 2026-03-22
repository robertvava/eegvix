"""Fine-tune Stable Diffusion with LoRA for EEG-conditioned generation.

Usage:
    python scripts/train_diffusion.py --checkpoint path/to/contrastive_checkpoint.ckpt
"""

import argparse

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from eegvix.utils.seed import seed_everything
from eegvix.data.datamodule import ThingsEEG2DataModule
from eegvix.training.diffusion_module import DiffusionFineTuneModule


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Contrastive training checkpoint")
    parser.add_argument("--data_dir", type=str, default="eeg_dataset")
    parser.add_argument("--sd_model", type=str, default="stabilityai/stable-diffusion-2-1")
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)

    datamodule = ThingsEEG2DataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=4,
        average_repetitions=True,
    )

    model = DiffusionFineTuneModule(
        eeg_encoder_checkpoint=args.checkpoint,
        sd_model=args.sd_model,
        lora_rank=args.lora_rank,
        learning_rate=args.lr,
    )

    callbacks = [
        ModelCheckpoint(monitor="train/diffusion_loss", mode="min", save_top_k=3),
    ]

    logger = WandbLogger(project="eegvix-v2-diffusion")

    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        precision="16-mixed",
        gradient_clip_val=1.0,
        callbacks=callbacks,
        logger=logger,
    )

    trainer.fit(model, datamodule)


if __name__ == "__main__":
    main()
