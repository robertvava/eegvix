# EEGVIX

Using autoencoders and generative models for generating ground truth visual stimuli (images) from EEG Signals (brainwaves). 

This project has been submitted as part of the MSc in Artificial Intelligence and Adaptive Systems from University of Sussex, under the supervision of Dr. Ivor Simpson. 

## Data 
The dataset required for running this experiment can be found at: https://doi.org/10.1016/j.neuroimage.2022.119754. 

## Running experiments
You can create the conda environment using: 
```
conda env create -f envinronment.yml
```
and activate it with: 
```
conda activate eeg_vis
```

You can run the experiment by executing 
```
python main.py --model_name vae --num_epochs 500 --mode train --batch_size 32
```

You will be prompted for a Weights and Biases (wandb) setup as such: 
```
wandb: (1) Create a W&B account
wandb: (2) Use an existing W&B account
wandb: (3) Don't visualize my results
wandb: Enter your choice:
```

If you have no wandb account, you can simply press 3 (no visualization) and enter. 

More config parameters can be tweaked in config.py (inclusive of hyperparameters) as follows: 
```
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
```

## Results 

Below you can find the overall results of the projects. The results of the diffusion model have been obtained after the submission of the dissertation, which is why they have not been included. 

### Aligned Recon
The results of the Joint/Aligned model reconstruction: 

![Aligned Reconstruction](https://github.com/robertvava/eegvix/blob/main/assets/align_rec.png)

### Concept Space
The task of image recreation from brain signals, without an intermediary kernel, is a difficult one - in previous research, it has been done using an EEG -> String description -> BERT -> Reconstructed Image, or other intermediary methods. The success of this project is defined by the best performing model identifying a broad similarity space, which encapsulates both concepts, and image characteristics: 

![Concept Space](https://github.com/robertvava/eegvix/blob/main/assets/concept_space.png)

### Reconstruction tasks

A secondary achievement was the autoencoding tasks for reconstructing the EEGs and Images, depicted below: 

#### Image Reconstruction
![Image Reconstructions](https://github.com/robertvava/eegvix/blob/main/assets/image_rec.png)

#### EEG Reconstructions 
![EEG Reconstructions](https://github.com/robertvava/eegvix/blob/main/assets/eeg_rec.png)

## Contact 

For results, discussions, or seeing the written thesis, or anything else, please contact me at vavarobert10@gmail.com.

If you would like to use the pipeline on a custom dataset, with your own models, please feel free to do so! 
