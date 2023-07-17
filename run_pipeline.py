from collate_utils import *
from misc_utils import *
from dataloaders.load_dl import *
import wandb

def run_pipeline(model_config):

    if model_config['act'] == 'train':

        g_cpu = torch.Generator()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        """If reduced, yet to be implemented. """

        idx_val = validation_strat()
        limit_samples = limit_samples(idx_val)
        all_idx = all_idx(idx_val, limit_samples)
        X_train, X_val, X_test, chnames, times = collate_participant_eeg(idx_val, reduce = True, limit_samples = limit_samples, all_idx = all_idx, to_torch = True)
        y_train, y_val, y_test = collate_images_dataset(idx_val, reduce = True, limit_samples = limit_samples)

        """Full data.   """

        # X_train, X_val, X_test, chnames, times = collate_participant_eeg(idx_val, to_torch = True)
        # y_train, y_val, y_test = collate_images_dataset(idx_val)
        
        


        train_dl, val_dl, test_dl = create_dataloaders(g_cpu, X_train, X_val, X_test, y_train, y_val, y_test)

       
    elif model_config['act'] == 'generate':
        return 1

def train(config):
    return 0
