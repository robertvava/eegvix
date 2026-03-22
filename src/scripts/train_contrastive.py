"""Train the EEG encoder via contrastive alignment with CLIP.

Usage:
    python scripts/train_contrastive.py
    python scripts/train_contrastive.py +experiment=debug
    python scripts/train_contrastive.py training.max_epochs=100 data.batch_size=128
"""

import hydra
from omegaconf import DictConfig
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger

from eegvix.utils.seed import seed_everything
from eegvix.data.datamodule import ThingsEEG2DataModule
from eegvix.training.contrastive_module import ContrastiveAlignmentModule
from eegvix.training.callbacks import EmbeddingVisualizationCallback, RetrievalAccuracyCallback


@hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)

    # Data
    datamodule = ThingsEEG2DataModule(
        data_dir=cfg.data.data_dir,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        average_repetitions=cfg.data.average_repetitions,
        val_n_concepts=cfg.data.val_n_concepts,
        val_random_state=cfg.data.val_random_state,
        clip_embeddings_path=f"{cfg.data.clip_embeddings_dir}/train_clip_embeddings.pt",
    )

    # Model
    model = ContrastiveAlignmentModule(
        n_channels=cfg.data.n_channels,
        n_timepoints=cfg.data.n_timepoints,
        embed_dim=cfg.model.eeg_encoder.embed_dim,
        num_temporal_conv_layers=cfg.model.eeg_encoder.num_temporal_conv_layers,
        temporal_kernel_sizes=list(cfg.model.eeg_encoder.temporal_kernel_sizes),
        n_spatial_heads=cfg.model.eeg_encoder.n_spatial_heads,
        n_temporal_transformer_layers=cfg.model.eeg_encoder.n_temporal_transformer_layers,
        n_temporal_heads=cfg.model.eeg_encoder.n_temporal_heads,
        dropout=cfg.model.eeg_encoder.dropout,
        use_frequency_branch=cfg.model.eeg_encoder.use_frequency_branch,
        output_dim=cfg.model.eeg_encoder.output_dim,
        n_subjects=cfg.data.n_subjects,
        proj_hidden_dim=cfg.model.projection_head.hidden_dim,
        init_temperature=cfg.training.init_temperature,
        learnable_temperature=cfg.training.learnable_temperature,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        warmup_epochs=cfg.training.warmup_epochs,
        max_epochs=cfg.training.max_epochs,
        subject_embedding_warmup_epochs=cfg.training.subject_embedding_warmup_epochs,
        clip_model_name=cfg.model.clip.model_name,
        clip_pretrained=cfg.model.clip.pretrained,
        use_precomputed_clip=True,
    )

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            monitor="val/loss",
            mode="min",
            save_top_k=3,
            filename="epoch={epoch}-val_loss={val/loss:.4f}",
            auto_insert_metric_name=False,
        ),
        EarlyStopping(
            monitor="val/loss",
            patience=cfg.training.early_stopping_patience,
            mode="min",
        ),
        LearningRateMonitor(logging_interval="epoch"),
        RetrievalAccuracyCallback(top_k=[1, 5, 10], every_n_epochs=5),
        EmbeddingVisualizationCallback(every_n_epochs=10),
    ]

    # Logger
    logger = WandbLogger(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        tags=cfg.wandb.tags,
        mode=cfg.wandb.mode,
    )

    # Trainer
    trainer = L.Trainer(
        max_epochs=cfg.training.max_epochs,
        precision=cfg.training.precision,
        gradient_clip_val=cfg.training.gradient_clip_val,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        check_val_every_n_epoch=cfg.training.check_val_every_n_epoch,
        callbacks=callbacks,
        logger=logger,
        deterministic=True,
    )

    trainer.fit(model, datamodule)
    trainer.test(model, datamodule)


if __name__ == "__main__":
    main()
