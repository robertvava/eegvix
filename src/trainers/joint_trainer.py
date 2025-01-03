import torch.nn as nn 
from torch.optim import Adam
from config import HPConfig, ExperimentConfig
from torch.utils.data import DataLoader
import torch.optim
from misc_utils import denormalize
import wandb
import matplotlib.pyplot as plt
from torchvision.transforms import functional as Fnc
from models.joint_model.joint_model import *

hp = HPConfig()
config = ExperimentConfig()


class JointTrainer:
    def __init__(self, latent_dim:int = 512, visualise = False, device: torch.device = 'cpu', save_model = False):
        self.latent_dim = latent_dim
        self.visualise = visualise
        self.save_model = save_model

    def train(self, train_dl: DataLoader, val_dl: DataLoader, epoch: int, device: torch.device, latent_dim:int = 512):
    disc_cirterion = nn.BCELoss()  
    criterion = ReconLoss()
    
    eeg_encoder = EEGEncoder(latent_dim = latent_dim).to(device)
    image_encoder = ImageEncoder(latent_dim = latent_dim).to(device)
    image_decoder = ImageDecoder(latent_dim = latent_dim).to(device)
    
    eeg_encoder_config = torch.load('trained_models/best_aligned_eeg_encoder' + str(latent_dim) + '.pt')
    image_encoder_config = torch.load('trained_models/best_aligned_image_encoder' + str(latent_dim) + '.pt')
    image_decoder_config = torch.load('trained_models/best_img_decoder' + str(latent_dim) + '.pt')
    
    eeg_encoder.load_state_dict(eeg_encoder_config)
    image_encoder.load_state_dict(image_encoder_config)
    image_decoder.load_state_dict(image_decoder_config)
    
    learning_rate = 0.00005
    lrD = 0.0005  
    real_label = 0.9
    fake_label = 0.05
    noise_dim = 100
    num_epochs = 1000
    early_stopping_patience = 25
    epochs_without_improvement = 0
    discriminator = Discriminator().to(device)
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=lrD, weight_decay = 1e-5)
    disc_scheduler = ReduceLROnPlateau(disc_optimizer, 'min', patience=10, factor=0.5, verbose=True)
    lambda_gp = 10
    
    best_val_loss = float('inf')
    optimizer = optim.Adam(image_decoder.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)

    for epoch in range(num_epochs):
        eeg_encoder.train()
        image_decoder.train()
        discriminator.train()
        train_loss = 0

        for batch_idx, (real_eegs, real_imgs) in enumerate(train_dl):
            real_eegs, real_imgs = real_eegs.to(device), real_imgs.to(device)
            
            eeg_latent = eeg_encoder(real_eegs)
            img_latent = image_encoder(real_imgs)
            batch_size = real_imgs.size(0)
            
            # Discriminator 
            fake_images = image_decoder(eeg_latent).detach()  
            label = torch.full((batch_size,), real_label, device=device).float()
            outputs_real = discriminator(real_imgs)
            d_loss_real = -outputs_real.mean()
            outputs_fake = discriminator(fake_images)
            d_loss_fake = outputs_fake.mean()
            gp = gradient_penalty(discriminator, eeg_latent, real_imgs, fake_images, device=device)
            d_loss = d_loss_real + d_loss_fake + lambda_gp * gp

            disc_optimizer.zero_grad()
            optimizer.zero_grad()
            outputs = discriminator(fake_images)
            outputs = image_decoder(eeg_latent)
            
            loss = criterion(outputs, real_imgs) + 0.5 * d_loss
            
            loss.backward()
            optimizer.step()
            disc_optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_dl)

        eeg_encoder.eval()
        image_decoder.eval()
        discriminator.eval()
        total_val_loss = 0

        for val_real_eegs, val_real_imgs in val_dl:
            val_real_eegs, val_real_imgs = val_real_eegs.to(device), val_real_imgs.to(device)
            
            val_eeg_latent = eeg_encoder(val_real_eegs)
            val_img_latent = image_encoder(val_real_imgs)
            val_fake_imgs = image_decoder(val_eeg_latent)
            #Discriminator 
            batch_size = val_real_imgs.size(0)
            val_label = torch.full((batch_size,), real_label, device=device).float()
            val_outputs_real = discriminator(val_real_imgs)
            val_d_loss_real = -val_outputs_real.mean()
            val_noise = torch.randn(val_real_imgs.size(0), noise_dim).to(device)
            val_outputs_fake = discriminator(val_fake_imgs.detach())
            val_d_loss_fake = val_outputs_fake.mean()
            val_gp = gradient_penalty(discriminator, val_eeg_latent, val_real_imgs, val_fake_imgs, device=device)
            val_recon_imgs = discriminator(val_fake_imgs)
            val_d_loss = val_d_loss_real + val_d_loss_fake + lambda_gp * val_gp

            val_loss = criterion(val_fake_imgs, val_real_imgs) + 0.5 * val_d_loss
            total_val_loss += loss.item()

        total_val_loss /= len(val_dl)

        scheduler.step(val_loss)
        disc_scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            if self.save_model:
                torch.save({'joint_eeg_encoder': eeg_encoder.state_dict(),
                            'joint_image_encoder': image_encoder.state_dict(),
                            'joint_image_decoder': image_decoder.state_dict(),
                            'joint_discriminator': discriminator.state_dict()}, 'trained_models/joint_model' + str(latent_dim) + '.pt')

        else:
            epochs_without_improvement += 1
            if epochs_without_improvement == early_stopping_patience:
                print("Early stopping!")
                break

        if epoch % 5 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {loss.item():.4f}, Validation Loss: {val_loss:.4f}")
            if self.visualise:
                fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))  

        
                axes[0, 0].imshow(Fn.to_pil_image(denormalize(real_imgs[5], mean, std)))
                axes[0, 0].set_title('Real Training Image')
                axes[0, 0].axis('off')  


                axes[0, 1].imshow(Fn.to_pil_image(denormalize(outputs[5], mean, std)))
                axes[0, 1].set_title('Reconstructed Training Image')
                axes[0, 1].axis('off')

                axes[1, 0].imshow(Fn.to_pil_image(denormalize(val_real_imgs[5], mean, std)))  #
                axes[1, 0].set_title('Real Validation Image')
                axes[1, 0].axis('off')

                axes[1, 1].imshow(Fn.to_pil_image(denormalize(val_fake_imgs[5], mean, std)))
                axes[1, 1].set_title('Reconstructed Validation Image')
                axes[1, 1].axis('off')

                plt.tight_layout()
                plt.show()