from collate_utils import collate_images_dataset, collate_participant_eeg
from misc_utils import get_limit_samples, get_validation_strat, get_all_idx
from dataloaders.load_dl import *
import torch.optim as optim
import torch
import wandb
import torch.nn as nn
from models.no_gen.logreg import RegressionModel
import os

def run_pipeline(model_config, reduce = False):

    if model_config['act'] == 'train':

        g_cpu = torch.Generator()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        idx_val = get_validation_strat() # Validation idx
        
        """If reduced, yet to be implemented. """
        if reduce: 

            limit_samples = get_limit_samples(idx_val)
            all_idx = get_all_idx(idx_val, limit_samples)
            X_train, X_val, X_test, chnames, times = collate_participant_eeg(idx_val, reduce = True, limit_samples = limit_samples, all_idx = all_idx, to_torch = True)
            y_train, y_val, y_test = collate_images_dataset(idx_val, reduce = True, limit_samples = limit_samples)

        # Full data
        else: 
            X_train, X_val, X_test, chnames, times = collate_participant_eeg(idx_val, to_torch = True)
            y_train, y_val, y_test = collate_images_dataset(idx_val)
        
        # if os.path.isfile('dataloaders/train_dl.pt'):
        #     train_dl = torch.load('dataloaders/train_dl.pt')
        #     val_dl = torch.load('dataloaders/val_dl.pt')
        #     test_dl = torch.load('dataloaders/test_dl.pt')
        # else: 
        train_dl, val_dl, test_dl = create_dataloaders(g_cpu, X_train, X_val, X_test, y_train, y_val, y_test)
            # torch.save(train_dl, 'dataloaders/train_dl.pt')
            # torch.save(val_dl, 'dataloaders/val_dl.pt')
            # torch.save(test_dl, 'dataloaders/test_dl.pt')

        train(train_dl, model_config, device)

    elif model_config['act'] == 'generate':
        return 1

def train(train_dl, config, device):
    print ("Model name: ", config['model'])
    if config['model'] == 'reg':
        model = RegressionModel().to(device)
    # model = RegressionModel().to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        model.train() # Sets the module in training mode. 
        for epoch in range(config['epochs']):
            for i, (inputs, targets) in enumerate(train_dl):
                inputs = inputs.to(device)
                targets = targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                if epoch % 5 == 0:
                    print(f"Epoch [{epoch+1}/{config['epochs']}], Batch [{i+1}/{len(train_dl)}], Loss: {loss.item():.4f}")

    elif config['model'] == 'vae':
        from models.VAE.vae import VAE, VAELoss
        
        latent_dim = 128
        learning_rate = 0.001
        num_epochs = 100
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create VAE, loss function, and optimizer
        model = VAE(latent_dim).to(device)
        loss_fn = VAELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        model.train()
        # Main training loop
        for epoch in range(num_epochs):
            
            running_loss = 0.0
            for i, (inputs, targets) in enumerate(train_dl):
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                optimizer.zero_grad()
                x_hat, mu, logvar = model(inputs)
                
                loss = loss_fn(x_hat, targets, mu, logvar)
                loss.backward()
                
                optimizer.step()
                running_loss += loss.item()

            average_loss = running_loss / len(train_dl)
            
            print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {average_loss:.4f}")