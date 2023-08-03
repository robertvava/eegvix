from dataloading_utils.main_load import get_dataloaders
import torch.optim as optim
import torch
import wandb
import torch.nn as nn
from models.no_gen.logreg import RegressionModel
import os
import numpy as np
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from torchvision.transforms import functional as F
from misc_utils import denormalize
from trainers.reg_trainer import NoGenTrainer


def run_pipeline(config):
    print ("Model name: ", config.model_name)
    print ("Mode: ", config.act)
    
    if config.act == 'train':
        wandb.init(
                # set the wandb project where this run will be logged
                project="eeg-vix",
                # track hyperparameters and run metadata
                config={
                "learning_rate": config.learning_rate,
                "architecture": "LogReg",
                "dataset": "Large and rich eeg-image dataset",
                "epochs": config.num_epochs,
                }
            )
        
        g_cpu = torch.Generator()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_dl, val_dl, test_dl = get_dataloaders(g_cpu, eeg_norm = config.eeg_norm, apply_mean = config.apply_mean, resolution = config.resolution, batch_size = config.batch_size)

        reg_trainer = NoGenTrainer()
        reg_trainer.train(train_dl, val_dl, config.num_epochs, device = device)
    
    elif config.act == 'generate':
        return 1



    
    
