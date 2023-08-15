
from models.no_gen.logreg import RegressionModel
from torch import optim as optim
from config import HPConfig, ExperimentConfig
from torch.utils.data import DataLoader
import torch.optim
from misc_utils import denormalize
import wandb
import matplotlib.pyplot as plt
from torchvision.transforms import functional as Fnc
from models.autoencoders.img_ae import ImageDecoder, ImageEncoder, IMGAELoss
from torchvision.transforms import functional as Fn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

hp = HPConfig()
config = ExperimentConfig()


class Img_AE_Trainer:
    def __init__(self, latent_dim:int = 512, resolution = 64, visualise = False, device: torch.device = 'cpu'):
        self.visualise = visualise
        self.latent_dim = latent_dim
        self.resolution = resolution
        self.encoder = ImageEncoder(input_channels=3, latent_dim=self.latent_dim)
        self.decoder = ImageDecoder(latent_dim=self.latent_dim, resolution=self.resolution)

        

    def train(self, train_dl: DataLoader, val_dl: DataLoader, device: torch.device, epochs:int  = 500):        

        latent_dim = self.latent_dim
        resolution = self.resolution  
        encoder = ImageEncoder(input_channels=3, latent_dim=latent_dim).to(device)
        decoder = ImageDecoder(latent_dim=latent_dim, resolution=resolution).to(device)

        num_epochs = config.num_epochs
        early_stopping_patience = config.early_stopping_patience
        epochs_without_improvement = 0
        best_val_loss = float('inf')
        learning_rate = config.learning_rate

        criterion = IMGAELoss()

        optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate, weight_decay=config.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)

        for epoch in range(num_epochs):
            encoder.train()
            decoder.train()
            train_loss = 0

            for batch_idx, (_, inputs) in enumerate(train_dl):
                real_images = inputs.to(device)

                # Encoding
                latent = encoder(real_images)

                # Decoding
                outputs = decoder(latent)

                loss = criterion(outputs, real_images)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_dl)

            encoder.eval()
            decoder.eval()
            val_loss = 0

            with torch.no_grad():
                for _, inputs in val_dl:
                    val_real_images = inputs.to(device)
                    latent = encoder(val_real_images)
                    val_outputs = decoder(latent)
                    loss = criterion(val_outputs, val_real_images)
                    val_loss += loss.item()

            val_loss /= len(val_dl)

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement == early_stopping_patience:
                    print("Early stopping!")
                    break

            if epoch % 5 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {loss.item():.4f}, Validation Loss: {val_loss:.4f}")
                if self.visualise: 
                    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))  # 2x2 grid of plots

                    # Real Training Image
                    axes[0, 0].imshow(Fn.to_pil_image(denormalize(real_images[0], config.mean, config.std)))
                    axes[0, 0].set_title('Real Training Image')
                    axes[0, 0].axis('off')  # Hide axes for images

                    # Reconstructed Training Image
                    axes[0, 1].imshow(Fn.to_pil_image(denormalize(outputs[0], config.mean, config.std)))
                    axes[0, 1].set_title('Reconstructed Training Image')
                    axes[0, 1].axis('off')

                    # Assuming you have real validation images and their reconstructions
                    # Real Validation Image
                    axes[1, 0].imshow(Fn.to_pil_image(denormalize(val_real_images[0], config.mean, config.std)))  #
                    axes[1, 0].set_title('Real Validation Image')
                    axes[1, 0].axis('off')

                    # Reconstructed Validation Image
                    axes[1, 1].imshow(Fn.to_pil_image(denormalize(val_outputs[0], config.mean, config.std)))
                    axes[1, 1].set_title('Reconstructed Validation Image')
                    axes[1, 1].axis('off')

                    plt.tight_layout()
                    plt.show()


                

