from collate_utils import collate_images_dataset, collate_participant_eeg
from misc_utils import get_limit_samples, get_validation_strat, get_all_idx
from dataloaders.load_dl import *
import torch.optim as optim
import torch
import wandb
import torch.nn as nn
from models.no_gen.logreg import RegressionModel

def run_pipeline(model_config):

    if model_config['act'] == 'train':

        g_cpu = torch.Generator()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        """If reduced, yet to be implemented. """

        idx_val = get_validation_strat()
        limit_samples = get_limit_samples(idx_val)
        all_idx = get_all_idx(idx_val, limit_samples)
        X_train, X_val, X_test, chnames, times = collate_participant_eeg(idx_val, reduce = True, limit_samples = limit_samples, all_idx = all_idx, to_torch = True)
        y_train, y_val, y_test = collate_images_dataset(idx_val, reduce = True, limit_samples = limit_samples)

        """Full data.   """

        # X_train, X_val, X_test, chnames, times = collate_participant_eeg(idx_val, to_torch = True)
        # y_train, y_val, y_test = collate_images_dataset(idx_val)
        

        train_dl, val_dl, test_dl = create_dataloaders(g_cpu, X_train, X_val, X_test, y_train, y_val, y_test)

        train(train_dl, model_config, device)

    elif model_config['act'] == 'generate':
        return 1

def train(train_dl, config, device):
    if config['model'] == 'reg':
        model = RegressionModel()
    model = RegressionModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(config['epochs']):
        for i, (inputs, targets) in enumerate(train_dl):
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{config['epochs']}], Batch [{i+1}/{len(train_dl)}], Loss: {loss.item():.4f}")


