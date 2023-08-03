from dataclasses import dataclass, field
import torch
import torch.nn as nn
from pathlib import Path
import os

@dataclass
class HPConfig:
    criterion: torch.nn.modules.loss = nn.MSELoss()
    learning_rate: float = 1e-4
    weight_decay: float = 1e-3
    model: str = 'VAE'
        
    def __post_init__(self):
        if self.model == 'GAN':
            self.criterion = nn.BCELoss()

        elif self.model == 'VAE':

            self.criterion = nn.KLDivLoss(reduction = "none")

        elif self.model == 'diff':

            self.criterion = nn.MSELoss()
    # GAN HP
    
    # VAE HP
        
    # Diffusions HP
        
@dataclass
class ExperimentConfig:
    hp = HPConfig()
    # General 
    learning_rate: float = hp.learning_rate
    model_name: str = 'vae'
    act: str = 'train'
    validation_n_samples: int = 100
    batch_size: int = 16
    full_exp: bool = False
    transformation: None = None
    num_workers: int = 4
    random_seed: int = 13
    shuffle_dl: bool = True
    drop_last_dl: bool = False
    mode: str = 'train'
    num_epochs: int = 100
    
    # Images config
    transform_resolution: tuple = field(init=False)
    mean: list = field(init = False)
    std: list = field(init = False)
    normalize: bool = True
    
    # EEGs config
    apply_mean: bool = True
    all_participants: bool = False
    eeg_norm: bool = True
        
    # Paths
    data_dir: Path = os.getcwd() + '/eeg_dataset'
    images_dir: Path = '/images'
    training_images_dir: Path = '/training'
    eeg_dir: Path = '/eeg_dataset'
    
    def __post_init__(self):
        self.mean = [0.54094851, 0.49473587, 0.4383250]
        self.std = [0.27202466, 0.26261519, 0.2781496]
        self.resolution = (64, 64)
    
    
        
# seed = exp_config.random_seed