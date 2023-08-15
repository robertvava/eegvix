import torch
import numpy as np
from sklearn.utils import resample
import glob
import os



def align_eegs(X, apply_mean = False, all_participants = False, test = False):
    
    X =  torch.swapaxes(X, 0, 1)

    if test:
        if apply_mean:
            X = X.reshape(-1, 17, 100)
        else: 
            X = X.reshape(-1, 80, 17, 100)
            X = X.reshape(-1, 17, 100)
    else:
        if apply_mean:
            X = X.reshape(-1, 17, 100)
        else: 
            X = X.reshape(-1, 4, 17, 100)
            X = X.reshape(-1, 17, 100)

    return X


def validation_strat(n_samples = 150):

    train_img_concepts = np.arange(1654)
    img_per_concept = 10
    val_concepts = np.sort(resample(train_img_concepts, replace=False,
        n_samples=n_samples))
    idx_val = np.zeros((len(train_img_concepts)*img_per_concept), dtype=bool)
    for i in val_concepts:
        idx_val[i*img_per_concept:i*img_per_concept+img_per_concept] = True

    return idx_val


def paths_to_npy(path = os.getcwd() + '/eeg_dataset' + '/preprocessed', test_data = False):
    """
    :return: EEG Test paths, Training paths.
    """
    if test_data:
        return glob.glob(path + "/sub-**/*_test.npy", recursive=True)

    return glob.glob(path + "/sub-**/*_training.npy", recursive=True)

def paths_to_subjects(path = os.getcwd() + '/eeg_dataset' + '/preprocessed'):
    return glob.glob(path + "/sub-**", recursive = True)
