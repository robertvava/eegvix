
from models.no_gen.logreg import RegressionModel
import torch.nn as nn 
from torch.optim import Adam
from config import HPConfig, ExperimentConfig
from torch.utils.data import DataLoader
import torch.optim
from misc_utils import denormalize
import wandb
import matplotlib.pyplot as plt
from torchvision.transforms import functional as Fnc
from models.autoencoders.eeg_ae import EEGAutoEncoder, EEGAELoss


hp = HPConfig()
config = ExperimentConfig()

class EEG_AE_Trainer:
    def __init__(self, latent_dim:int = 512, visualise = False, device: torch.device = 'cpu', save_model = False):
        self.latent_dim = latent_dim
        self.visualise = visualise

    def train(self, train_dl: DataLoader, val_dl: DataLoader, epoch: int, device: torch.device):

            # Training
        from torchvision.transforms import functional as Fn
        import torch.nn.functional as F
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        from torch.optim.lr_scheduler import ReduceLROnPlateau

        num_epochs = 1000
        criterion = EEGAELoss()
        t = [i for i in range(100)]
        early_stopping_patience = config.early_stopping_patience
        epochs_without_improvement = 0
        latent_dim = self.latent_dim

        lambda_sparsity = 1e-5
        best_val_loss = float('inf')
        # patience = 5  # Number of epochs to wait before adjusting lambda_sparsity
        # counter = 0

        model = EEGAutoEncoder(latent_dim = latent_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr = config.learning_rate, weight_decay=config.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=25, factor=0.5, verbose=True)

        for epoch in range(num_epochs):
            model.train()

            for batch_idx, (inputs, _) in enumerate(train_dl):
                inputs = inputs.to(device)
                outputs = model(inputs)
                recon_loss = criterion(outputs, inputs)

                latent_representation = model.encoder(inputs)
                sparsity_penalty = torch.norm(latent_representation, 1)
                loss = recon_loss + lambda_sparsity * sparsity_penalty
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            
            model.eval()
            val_loss = 0

            with torch.no_grad():
                for batch_idx, (val_inputs, _) in enumerate(val_dl):  
                    val_inputs = val_inputs.to(device)
                    val_outputs = model(val_inputs)
                    recon_loss = criterion(val_outputs, val_inputs)
                    latent_representation = model.encoder(val_inputs)
                    sparsity_penalty = torch.norm(latent_representation, 1)
                    val_loss += recon_loss + lambda_sparsity * sparsity_penalty

            val_loss /= len(val_dl)  
            scheduler.step(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                if save_model:
                    torch.save(encoder.state_dict(), 'trained_models/best_img_encoder' + str(latent_dim) + '.pt')
                    torch.save(decoder.state_dict(), 'trained_models/best_img_decoder' + str(latent_dim) + '.pt')
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement == early_stopping_patience:
                    print("Early stopping!")
                    break

            if epoch % 5 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {loss.item():.4f}, Validation Loss: {val_loss:.4f}")

                if self.visualise: 
                    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))  # 2x2 grid of plots

                    # Real train EEG
                    axes[0, 0].plot(t, inputs[0][0].cpu())
                    axes[0, 0].set_yticks((-0.15, 1.15))
                    axes[0, 0].set_title('Real train EEG')

                    # Train EEG
                    axes[0, 1].plot(t, outputs[0][0].cpu().detach().numpy())
                    axes[0, 1].set_yticks((-0.15, 1.15))
                    axes[0, 1].set_title('Train EEG')

                    # Val real EEG
                    axes[1, 0].plot(t, val_inputs[0][0].cpu())
                    axes[1, 0].set_yticks((-0.15, 1.15))
                    axes[1, 0].set_title('Val real EEG')

                    # Val EEG
                    axes[1, 1].plot(t, val_outputs[0][0].cpu())
                    axes[1, 1].set_yticks((-0.15, 1.15))
                    axes[1, 1].set_title('Val EEG')
                    plt.show()


        #     if epoch % 25 == 0:  # Every 50 epochs
        #         latent_representation = model.encoder(inputs).cpu().detach().numpy()
        #         embedded = TSNE(n_components=3, random_state=42, learning_rate = 'auto', init = 'pca').fit_transform(latent_representation)
        #         plt.scatter(embedded[:, 0], embedded[:, 1])
        #         plt.title(f"Latent Space at Epoch {epoch+1}")
        #         plt.show()


