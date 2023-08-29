import torch.nn as nn 
from torch.optim import Adam
from config import HPConfig, ExperimentConfig
from torch.utils.data import DataLoader
import torch.optim
from misc_utils import denormalize
import wandb
import matplotlib.pyplot as plt
from torchvision.transforms import functional as Fnc
from models.alignment.alignment_model import *



hp = HPConfig()
config = ExperimentConfig()

class AlignmentTrainer:
    def __init__(self, latent_dim:int = 512, visualise = False, device: torch.device = 'cpu', save_model = False):
        self.latent_dim = latent_dim
        self.visualise = visualise
        self.save_model = save_model

    @staticmethod
    def plot_histograms(eeg_latents, img_latents):
        plt.figure(figsize=(12, 8))
        plt.hist(eeg_latents.flatten(), bins=50, alpha=0.5, label="EEG Latents")
        plt.hist(img_latents.flatten(), bins=50, alpha=0.5, label="Image Latents")
        plt.legend()
        plt.title("Histogram of Latent Values")
        plt.show()

    def train(self, train_dl: DataLoader, val_dl: DataLoader, epoch: int, device: torch.device):
        from torchvision.transforms import functional as Fn
        import torch.nn.functional as F
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        from torch.optim.lr_scheduler import ReduceLROnPlateau

        image_encoder = ImageEncoder(latent_dim = latent_dim).to(device)
        image_encoder.load_state_dict(torch.load('trained_models/best_img_encoder' + str(latent_dim) + '.pt'))

        eeg_encoder = EEGEncoder(17, latent_dim=latent_dim).to(device)
        eeg_encoder.load_state_dict(torch.load('trained_models/best_eeg_encoder' + str(latent_dim) + '.pt'))

        image_decoder = ImageDecoder(latent_dim = latent_dim).to(device)
        image_decoder.load_state_dict(torch.load('trained_models/best_img_decoder' + str(latent_dim) + '.pt'))
        
        best_val_loss = float('inf')
        
        eeg_encoder_optimizer = torch.optim.Adam(eeg_encoder.parameters(), lr=learning_rate)

        eeg_scheduler = ReduceLROnPlateau(eeg_encoder_optimizer, 'min', patience=10, factor=0.5, verbose=True)
        
        epochs_without_improvement = 0


    for epoch in range(num_epochs):
        eeg_encoder.train()

        alignment_train_loss = 0
        umap_eeg_latents = []
        umap_img_latents = []

        if epoch % 5 == 0:
            print (f"Now training at epoch: {epoch+1}/{num_epochs}")

        for batch_idx, (real_eegs, real_imgs) in enumerate(train_dl):
            real_eegs, real_imgs = real_eegs.to(device), real_imgs.to(device)

            eeg_latent = eeg_encoder(real_eegs)
            img_latent = image_encoder(real_imgs)
            outputs = image_decoder(eeg_latent)
            outputs_image_latent = image_decoder(img_latent)
            with torch.no_grad():
                umap_eeg_latent = eeg_encoder(real_eegs)
                umap_img_latent = image_encoder(real_imgs)
                umap_eeg_latents.append(umap_eeg_latent)
                umap_img_latents.append(umap_img_latent)

            image_encoder.zero_grad()
            eeg_encoder.zero_grad()

            negative_latent = get_semi_hard_negatives(eeg_latent, img_latent, margin = 0.25)
            loss = alignment_loss_criterion(eeg_latent, img_latent, negative_latent)

            loss.backward()

            eeg_encoder_optimizer.step()

            alignment_train_loss += loss.item()

        alignment_train_loss /= len(train_dl)


        image_encoder.eval()
        eeg_encoder.eval()

        total_val_loss = 0

        if epoch % 5 == 0:
            print (f"Now validating at epoch: {epoch+1}/{num_epochs}")

        for val_batch_idx, (real_val_eeg, real_val_imgs) in enumerate(val_dl):  
            real_val_eeg, real_val_imgs = real_val_eeg.to(device), real_val_imgs.to(device)        
            val_eeg_latent = eeg_encoder(real_val_eeg)
            val_img_latent = image_encoder(real_val_imgs)
            val_outputs = image_decoder(val_eeg_latent)
            val_outputs_image_latent = image_decoder(val_img_latent)
            val_negative_latent = get_semi_hard_negatives(val_eeg_latent, val_img_latent, margin = 0.25)
            val_alignment_loss = alignment_loss_criterion(val_eeg_latent, val_img_latent, val_negative_latent)
            total_val_loss += val_alignment_loss.item()

        total_val_loss /= len(val_dl)

        eeg_scheduler.step(total_val_loss)

        if total_val_loss < best_val_loss:
            best_val_loss = total_val_loss
            epochs_without_improvement = 0
            if self.save_model:
                torch.save(eeg_encoder.state_dict(), 'trained_models/best_aligned_eeg_encoder' + str(latent_dim) + '.pt')
                torch.save(image_encoder.state_dict(), 'trained_models/best_aligned_image_encoder' + str(latent_dim) + '.pt')

        else:
            epochs_without_improvement += 1
            if epochs_without_improvement == early_stopping_patience:
                print("Early stopping!")
                break

        if epoch % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {alignment_train_loss:.4f}, Validation Loss: {total_val_loss:.4f}")

            fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 10))  # 2x2 grid of plots

            # Real Training Image
            axes[0, 0].imshow(Fn.to_pil_image(denormalize(real_imgs[0], mean, std)))
            axes[0, 0].set_title('Real Training Image')
            axes[0, 0].axis('off')  # Hide axes for images

            # Reconstructed Training Image
            axes[0, 1].imshow(Fn.to_pil_image(denormalize(outputs[0], mean, std)))
            axes[0, 1].set_title('EEG Encoder Reconstruction')
            axes[0, 1].axis('off')
            
             # Reconstructed Training Image
            axes[0, 2].imshow(Fn.to_pil_image(denormalize(outputs_image_latent[0], mean, std)))
            axes[0, 2].set_title('Image Encoder Reconstruction')
            axes[0, 2].axis('off')


            # Assuming you have real validation images and their reconstructions
            # Real Validation Image
            axes[1, 0].imshow(Fn.to_pil_image(denormalize(real_val_imgs[0], mean, std)))  #
            axes[1, 0].set_title('Real Validation Image')
            axes[1, 0].axis('off')

            # Reconstructed Validation Image
            axes[1, 1].imshow(Fn.to_pil_image(denormalize(val_outputs[0], mean, std)))
            axes[1, 1].set_title('EEG Encoder Reconstruction')
            axes[1, 1].axis('off')
            
             # Reconstructed Validation Image
            axes[1, 2].imshow(Fn.to_pil_image(denormalize(val_outputs_image_latent[0], mean, std)))
            axes[1, 2].set_title('Image Encoder Reconstruction')
            axes[1, 2].axis('off')

            plt.tight_layout()
            plt.show()


            self.plot_histograms(umap_eeg_latents, umap_img_latents)
    