import numpy as np 
from sklearn.utils import resample
import glob
import os

def get_validation_strat(n_samples = 100):

    """
    :return: Positional IDs of chosen validation samples. 
    These IDs will be used to create a validation set from the training set. 
    """
    train_img_concepts = np.arange(1654)
    img_per_concept = 10
    val_concepts = np.sort(resample(train_img_concepts, replace=False,
        n_samples=n_samples))
    idx_val = np.zeros((len(train_img_concepts)*img_per_concept), dtype=bool)
    for i in val_concepts:
        idx_val[i*img_per_concept:i*img_per_concept+img_per_concept] = True

    return idx_val

def get_limit_samples(idx_val, n_samples = 50):
    """
    Reduce the dataset to n_samples*10 samples total. 
    :return: Reduced dataset samples indices. 
    """
    
    train_img_concepts = np.arange(1654)
    img_per_concept = 10
    val_concepts = np.sort(resample(train_img_concepts, replace=False,
        n_samples=n_samples))
    
    samples = np.zeros((len(train_img_concepts)*img_per_concept), dtype=bool)
    for i in val_concepts:
        if idx_val[i*img_per_concept:i*img_per_concept+img_per_concept].all() == True:
            pass
        else: 
            samples[i*img_per_concept:i*img_per_concept+img_per_concept] = True
    return samples

def get_all_idx(idx_val, limit_samples):
    
    train_img_concepts = np.arange(1654)
    img_per_concept = 10
    all_idx = np.zeros((len(train_img_concepts)*img_per_concept), dtype=bool)
    
    for (i, d) in enumerate(idx_val): 
        if d == True: 
            all_idx[i] = True
            
    for (i, d) in enumerate(limit_samples): 
        if d == True: 
            all_idx[i] = True
    
    return all_idx

def paths_to_npy(path, test_data = False):
    """
    :return: EEG Test paths, Training paths.
    """
    if test_data:
        return glob.glob(path + "/sub-**/*_test.npy", recursive=True)

    return glob.glob(path + "/sub-**/*_training.npy", recursive=True)

def paths_to_subjects(path = os.getcwd() + '/eeg_dataset' + '/preprocessed'):
    return glob.glob(path + "/sub-**", recursive = True)