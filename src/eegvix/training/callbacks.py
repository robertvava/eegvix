"""Custom Lightning callbacks for training visualization and monitoring."""

import torch
import lightning as L
import wandb
import numpy as np


class EmbeddingVisualizationCallback(L.Callback):
    """Log UMAP/t-SNE projections of EEG and image embeddings to wandb."""

    def __init__(self, every_n_epochs: int = 10, max_samples: int = 500):
        self.every_n_epochs = every_n_epochs
        self.max_samples = max_samples

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if trainer.current_epoch % self.every_n_epochs != 0:
            return

        try:
            from sklearn.manifold import TSNE
        except ImportError:
            return

        eeg_embeds = []
        img_embeds = []
        image_ids = []

        with torch.no_grad():
            for batch in trainer.val_dataloaders:
                eeg = batch["eeg"].to(pl_module.device)
                subject_ids = batch["subject_id"].to(pl_module.device)

                eeg_embed = pl_module(eeg, subject_ids)
                eeg_embeds.append(eeg_embed.cpu())

                if "clip_embedding" in batch:
                    img_embeds.append(batch["clip_embedding"])
                image_ids.extend(batch["image_id"].tolist())

                if len(eeg_embeds) * eeg.size(0) >= self.max_samples:
                    break

        eeg_all = torch.cat(eeg_embeds, dim=0)[:self.max_samples].numpy()
        if img_embeds:
            img_all = torch.cat(img_embeds, dim=0)[:self.max_samples].numpy()
            combined = np.concatenate([eeg_all, img_all], axis=0)
            labels = ["eeg"] * len(eeg_all) + ["image"] * len(img_all)
        else:
            combined = eeg_all
            labels = ["eeg"] * len(eeg_all)

        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(combined) - 1))
        projected = tsne.fit_transform(combined)

        table = wandb.Table(columns=["x", "y", "type"])
        for (x, y), label in zip(projected, labels):
            table.add_data(float(x), float(y), label)

        wandb.log({
            "embedding_space": wandb.plot.scatter(
                table, "x", "y", groupKeys="type",
                title=f"Embedding Space (epoch {trainer.current_epoch})"
            )
        })


class RetrievalAccuracyCallback(L.Callback):
    """Compute top-k retrieval accuracy on the validation set."""

    def __init__(self, top_k: list[int] | None = None, every_n_epochs: int = 5):
        self.top_k = top_k or [1, 5, 10]
        self.every_n_epochs = every_n_epochs

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if trainer.current_epoch % self.every_n_epochs != 0:
            return

        eeg_embeds = []
        img_embeds = []

        with torch.no_grad():
            for batch in trainer.val_dataloaders:
                eeg = batch["eeg"].to(pl_module.device)
                subject_ids = batch["subject_id"].to(pl_module.device)

                eeg_embed = pl_module(eeg, subject_ids)
                eeg_embeds.append(eeg_embed.cpu())

                if "clip_embedding" in batch:
                    img_embeds.append(batch["clip_embedding"])

        if not img_embeds:
            return

        eeg_all = torch.cat(eeg_embeds, dim=0)
        img_all = torch.cat(img_embeds, dim=0)

        # Cosine similarity matrix
        similarity = eeg_all @ img_all.T
        labels = torch.arange(similarity.size(0))

        for k in self.top_k:
            if k > similarity.size(1):
                continue
            top_k_preds = similarity.topk(k, dim=1).indices
            correct = (top_k_preds == labels.unsqueeze(1)).any(dim=1).float().mean()
            pl_module.log(f"val/top{k}_acc", correct, sync_dist=True)
