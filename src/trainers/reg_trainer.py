
from models.no_gen.logreg import RegressionModel
import torch
import torch.nn as nn
from torch.optim import Adam
from config import HPConfig, ExperimentConfig
from torch.utils.data import DataLoader
from misc_utils import denormalize
import wandb
import matplotlib.pyplot as plt
from torchvision.transforms import functional as Fnc

hp = HPConfig()
config = ExperimentConfig()

class RegTrainer:
    def __init__(self, criterion: str = 'mse', optimizer: str = 'adam', device: torch.device = 'cpu'):
        self.model = RegressionModel()
        self.criterion = criterion
        self.optimizer = optimizer

        if criterion == 'mse':
            self.criterion = nn.MSELoss()
        elif criterion == 'l1':
            self.criterion = nn.L1Loss(reduction='none')

        if optimizer == 'adam':
            self.optimizer = Adam(self.model.parameters(),
                                lr=hp.learning_rate,
                                weight_decay=hp.weight_decay,
                                )

    def train(self, train_dl: DataLoader, val_dl: DataLoader, num_epochs: int, device: torch.device, epochs: int = 500):

        model = self.model
        criterion = self.criterion
        optimizer = self.optimizer
        num_epochs = config.num_epochs
        model.to(device)

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0

            for inputs, targets in train_dl:
                inputs = inputs.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            epoch_loss = running_loss / len(train_dl)
            wandb.log({"loss": epoch_loss})
            if epoch % 500 == 0:
                plt.figure()
                plt.imshow(Fnc.to_pil_image(denormalize(targets[13], config.mean, config.std)))
                plt.show()
                plt.figure()
                plt.imshow(Fnc.to_pil_image(denormalize(outputs[13], config.mean, config.std)))
                plt.show()
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.5f}")
