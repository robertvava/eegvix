from dataloading_utils.main_load import get_dataloaders
import torch
import wandb
import torch.nn as nn
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from torchvision.transforms import functional as F
from misc_utils import denormalize


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
                "architecture": config.model_name,
                "dataset": "Large eeg-image dataset",
                "epochs": config.num_epochs,
                }
            )
        
        g_cpu = torch.Generator()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_dl, val_dl, test_dl = get_dataloaders(g_cpu, eeg_norm = config.eeg_norm, apply_mean = config.apply_mean, resolution = config.resolution, batch_size = config.batch_size)

        if config.model_name == 'reg':
            from models.no_gen.logreg import RegressionModel
            from trainers.reg_trainer import RegTrainer
            reg_trainer = RegTrainer()
            reg_trainer.train(train_dl, val_dl, config.num_epochs, device = device, epochs = config.num_epochs)

        elif config.model_name == 'img_ae':
            from trainers.img_ae_trainer import Img_AE_Trainer
            trainer = Img_AE_Trainer(latent_dim = config.latent_dim, resolution = config.resolution[0])
            trainer.train(train_dl, val_dl, config.num_epochs, device = device, epochs = config.num_epochs)
        
        elif config.model_name == 'eeg_ae':
            from trainers.eeg_ae_trainer import EEG_AE_Trainer
            trainer = EEG_AE_Trainer(latent_dim = config.latent_dim, resolution = config.resolution[0])
            trainer.train(train_dl, val_dl, config.num_epochs, device = device, epochs = config.num_epochs)

        elif config.model_name == 'alignment':
            from trainers.alignment_trainer import AlignmentTrainer
            trainer = AlignmentTrainer(latent_dim = config.latent_dim, resolution = config.resolution[0])
            trainer.train(train_dl, val_dl, config.num_epochs, device = device, epochs = config.num_epochs, save_model = config.save_models)

        elif config.model_name = 'joint':
            from trainers.joint_trainer import JointTrainer
            trainer = JointTrainer(latent_dim = config.latent_dim, resolution = config.resolution[0])
            trainer.train(train_dl, val_dl, config.num_epochs, device = device, epochs = config.num_epochs, save_model = config.save_models)

    elif config.act == 'generate':
        return 1



    
    
