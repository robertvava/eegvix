import numpy as np 
import os 
from PIL import Image
from torchvision import transforms
from processing.img.pre.transf import transformation
from tqdm import tqdm
import torch

def collate_images_dataset(idx_val, reduce = False, limit_samples = False, data_dir = '/eeg_dataset', transform = None):

    img_dirs = os.path.join(os.getcwd() + data_dir, 'images', 'training_images')

    image_paths = []
    for root, dirs, files in os.walk(img_dirs):
        for file in files:
            if file.endswith(".jpg"):
                image_paths.append(os.path.join(root,file))
                
    image_paths.sort()

    if reduce: 
        reduced = []
    train = []
    val = []
    for i, image in enumerate(tqdm(image_paths, desc = 'Training and validation images loading...')):
        if idx_val[i] == True:
            img = Image.open(image).convert('RGB')
            img = transformation(img)
            val.append(img)
        elif reduce:
            if limit_samples[i] == True: 
                img = Image.open(image).convert('RGB')
                img = transformation(img)
                reduced.append(img)
        else:
            img = Image.open(image).convert('RGB')
            img = transformation(img)
            train.append(img)

    img_dirs = os.path.join(os.getcwd() + data_dir, 'images', 'test_images')

    image_paths = []

    for root, dirs, files in os.walk(img_dirs):
        for file in files:
            if file.endswith(".jpg"):
                image_paths.append(os.path.join(root,file))

    image_paths.sort()
    test = []
    for image in tqdm(image_paths, desc = 'Test images loading...'):
        img = Image.open(image).convert('RGB')
        img = transformation(img)
        test.append(img)

    if reduce: 
        return reduced, val, test
    
    return train, val, test

def collate_participant_eeg(idx_val, reduce = False, limit_samples = False, all_idx = False, participant = None, data_path ='eeg_dataset/preprocessed', sub = '/sub-02', to_torch = False):

    if participant:
        train_file = np.load(participant + '/preprocessed_eeg_training.npy', allow_pickle = True).item()
        test_file = np.load(participant + '/preprocessed_eeg_test.npy', allow_pickle = True).item()
    else:
        train_file = np.load(data_path + sub + '/preprocessed_eeg_training.npy', allow_pickle = True).item()
        test_file = np.load(data_path + sub + '/preprocessed_eeg_test.npy', allow_pickle = True).item()

    print (data_path + sub + '/preprocessed_eeg_training.npy')

    # Train and val
    signal_data = train_file['preprocessed_eeg_data']
    chnames = train_file['ch_names']
    times = train_file['times']
    signal_data = np.mean(signal_data, 1)
    if reduce: 
        signal_data_reduced = signal_data[limit_samples]
        signal_data_val = signal_data[idx_val]
        signal_data = np.delete(signal_data, all_idx, 0)
    else: 
        signal_data_val = signal_data[idx_val]
        signal_data = np.delete(signal_data, idx_val, 0)


    # Test
    signal_data_test = test_file['preprocessed_eeg_data']
    signal_data_test = np.mean(signal_data, 1)

    if to_torch:
        signal_data = torch.tensor(np.float32(signal_data))
        signal_data_val = torch.tensor(np.float32(signal_data_val))
        signal_data_test = torch.tensor(np.float32(signal_data))
        if reduce: 
            signal_data_reduced = torch.tensor(np.float32(signal_data_reduced))
        
    if reduce: 
        return signal_data_reduced, signal_data_val, signal_data_test, chnames, times
    return signal_data, signal_data_val, signal_data_test, chnames, times