from dataclasses import dataclass, field
import torch
import torch.nn as nn
from pathlib import Path
import os

# Hyperparameters configuration
@dataclass
class HPConfig:
    criterion: torch.nn.modules.loss = nn.MSELoss()
    learning_rate: float = 0.00005
    weight_decay: float = 1e-4
    latent_dim: int = 512

        
@dataclass
class ExperimentConfig:
    hp = HPConfig()

    # General 
    model_name: str = 'vae'
    act: str = 'train'
    batch_size: int = 32
    full_exp: bool = False
    transformation: None = None
    num_workers: int = 4
    random_seed: int = 13
    shuffle_dl: bool = True
    drop_last_dl: bool = False
    mode: str = 'train'
    num_epochs: int = 500
    save_model: bool = False
    
    # Images config
    transform_resolution: tuple = field(init=False)
    mean: list = field(init = False)
    std: list = field(init = False)
    normalize: bool = True
    
    # EEGs config
    apply_mean: bool = True
    all_participants: bool = False
    eeg_norm: bool = True
        
    # Experiment params
    learning_rate: float = hp.learning_rate
    validation_n_samples: int = 150
    latent_dim: int = hp.latent_dim
    early_stopping_patience: int = 25
    weight_decay:float = hp.weight_decay

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
