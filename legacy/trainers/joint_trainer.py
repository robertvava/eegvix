import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from config import HPConfig, ExperimentConfig
from torch.utils.data import DataLoader
from misc_utils import denormalize
import wandb
import matplotlib.pyplot as plt
from torchvision.transforms import functional as Fn
from models.joint_model.joint_model import (
    ImageEncoder, EEGEncoder, ImageDecoder, Discriminator
)
from models.autoencoders.img_ae import ReconLoss

hp = HPConfig()
config = ExperimentConfig()


def gradient_penalty(discriminator, real_imgs, fake_images, device):
    batch_size = real_imgs.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolated = (alpha * real_imgs + (1 - alpha) * fake_images).requires_grad_(True)
    d_interpolated = discriminator(interpolated)
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,
        retain_graph=True
    )[0]
    gradients = gradients.view(batch_size, -1)
    penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return penalty


class JointTrainer:
    def __init__(self, latent_dim: int = 512, visualise=False, device: torch.device = 'cpu', save_model=False):
        self.latent_dim = latent_dim
        self.visualise = visualise
        self.save_model = save_model

    def train(self, train_dl: DataLoader, val_dl: DataLoader, num_epochs: int, device: torch.device,
              epochs: int = 1000, save_model: bool = False):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        latent_dim = self.latent_dim
        mean = config.mean
        std = config.std

        criterion = ReconLoss()

        eeg_encoder = EEGEncoder(latent_dim=latent_dim).to(device)
        image_encoder = ImageEncoder(latent_dim=latent_dim).to(device)
        image_decoder = ImageDecoder(latent_dim=latent_dim).to(device)

        eeg_encoder.load_state_dict(torch.load('trained_models/best_aligned_eeg_encoder' + str(latent_dim) + '.pt'))
        image_encoder.load_state_dict(torch.load('trained_models/best_aligned_image_encoder' + str(latent_dim) + '.pt'))
        image_decoder.load_state_dict(torch.load('trained_models/best_img_decoder' + str(latent_dim) + '.pt'))

        # Freeze encoders — only fine-tune decoder with adversarial signal
        for param in eeg_encoder.parameters():
            param.requires_grad = False
        eeg_encoder.eval()

        for param in image_encoder.parameters():
            param.requires_grad = False
        image_encoder.eval()

        learning_rate = 0.00005
        lrD = 0.0005
        lambda_gp = 10
        early_stopping_patience = 25
        epochs_without_improvement = 0

        discriminator = Discriminator().to(device)

        # Separate optimizers for generator (decoder) and discriminator
        gen_optimizer = optim.Adam(image_decoder.parameters(), lr=learning_rate, weight_decay=1e-5)
        disc_optimizer = optim.Adam(discriminator.parameters(), lr=lrD, weight_decay=1e-5)
        gen_scheduler = ReduceLROnPlateau(gen_optimizer, 'min', patience=5, factor=0.5, verbose=True)
        disc_scheduler = ReduceLROnPlateau(disc_optimizer, 'min', patience=10, factor=0.5, verbose=True)

        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            image_decoder.train()
            discriminator.train()
            train_loss = 0

            for batch_idx, (real_eegs, real_imgs) in enumerate(train_dl):
                real_eegs, real_imgs = real_eegs.to(device), real_imgs.to(device)
                batch_size = real_imgs.size(0)

                with torch.no_grad():
                    eeg_latent = eeg_encoder(real_eegs)

                # --- Step 1: Update Discriminator ---
                fake_images = image_decoder(eeg_latent).detach()
                outputs_real = discriminator(real_imgs)
                outputs_fake = discriminator(fake_images)
                d_loss_real = -outputs_real.mean()
                d_loss_fake = outputs_fake.mean()
                gp = gradient_penalty(discriminator, real_imgs, fake_images, device=device)
                d_loss = d_loss_real + d_loss_fake + lambda_gp * gp

                disc_optimizer.zero_grad()
                d_loss.backward()
                disc_optimizer.step()

                fake_images_gen = image_decoder(eeg_latent)
                outputs_gen = discriminator(fake_images_gen)
                g_adv_loss = -outputs_gen.mean()
                g_recon_loss = criterion(fake_images_gen, real_imgs)
                g_loss = g_recon_loss + 0.5 * g_adv_loss

                gen_optimizer.zero_grad()
                g_loss.backward()
                gen_optimizer.step()

                train_loss += g_loss.item()

            train_loss /= len(train_dl)

            image_decoder.eval()
            discriminator.eval()
            total_val_loss = 0

            with torch.no_grad():
                for val_real_eegs, val_real_imgs in val_dl:
                    val_real_eegs, val_real_imgs = val_real_eegs.to(device), val_real_imgs.to(device)

                    val_eeg_latent = eeg_encoder(val_real_eegs)
                    val_fake_imgs = image_decoder(val_eeg_latent)

                    val_outputs_real = discriminator(val_real_imgs)
                    val_outputs_fake = discriminator(val_fake_imgs)
                    val_d_loss = -val_outputs_real.mean() + val_outputs_fake.mean()

                    val_recon_loss = criterion(val_fake_imgs, val_real_imgs)
                    val_loss = val_recon_loss + 0.5 * val_d_loss
                    total_val_loss += val_loss.item()

            total_val_loss /= len(val_dl)

            gen_scheduler.step(total_val_loss)
            disc_scheduler.step(total_val_loss)

            if total_val_loss < best_val_loss:
                best_val_loss = total_val_loss
                epochs_without_improvement = 0
                if save_model or self.save_model:
                    torch.save({
                        'joint_eeg_encoder': eeg_encoder.state_dict(),
                        'joint_image_encoder': image_encoder.state_dict(),
                        'joint_image_decoder': image_decoder.state_dict(),
                        'joint_discriminator': discriminator.state_dict()
                    }, 'trained_models/joint_model' + str(latent_dim) + '.pt')
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement == early_stopping_patience:
                    print("Early stopping!")
                    break

            if epoch % 5 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss:.4f}, Validation Loss: {total_val_loss:.4f}")
                wandb.log({"joint_train_loss": train_loss, "joint_val_loss": total_val_loss})

                if self.visualise:
                    with torch.no_grad():
                        outputs = image_decoder(eeg_encoder(real_eegs.to(device)))

                    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))

                    axes[0, 0].imshow(Fn.to_pil_image(denormalize(real_imgs[5].cpu(), mean, std)))
                    axes[0, 0].set_title('Real Training Image')
                    axes[0, 0].axis('off')

                    axes[0, 1].imshow(Fn.to_pil_image(denormalize(outputs[5].cpu(), mean, std)))
                    axes[0, 1].set_title('Reconstructed Training Image')
                    axes[0, 1].axis('off')

                    axes[1, 0].imshow(Fn.to_pil_image(denormalize(val_real_imgs[5].cpu(), mean, std)))
                    axes[1, 0].set_title('Real Validation Image')
                    axes[1, 0].axis('off')

                    axes[1, 1].imshow(Fn.to_pil_image(denormalize(val_fake_imgs[5].cpu(), mean, std)))
                    axes[1, 1].set_title('Reconstructed Validation Image')
                    axes[1, 1].axis('off')

                    plt.tight_layout()
                    plt.show()
