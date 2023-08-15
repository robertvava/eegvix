from torchvision import transforms
import os
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from sklearn.utils import resample
import numpy as np
import torch
from .misc_load import validation_strat, align_eegs


def collate_images_dataset(idx_val, resolution = (64, 64), data_dir = '/eeg_dataset'):

    transformation = transforms.Compose([
                transforms.Resize((resolution[0],resolution[1])),
                transforms.ToTensor(),
                transforms.Normalize([0.5409, 0.4947, 0.4383], [0.2720, 0.2626, 0.2781 ]),
                ])

    img_dirs = os.path.join(os.getcwd() + data_dir, 'images', 'training_images')

    image_paths = []
    for i, (root, dirs, files) in enumerate(os.walk(img_dirs)):
        for file in files:
            if file.endswith(".jpg"):
                image_paths.append(os.path.join(root,file))

    image_paths.sort()

    train = []
    val = []
    for i, image in enumerate(tqdm(image_paths, desc = 'Training and validation images loading...')):
        if idx_val[i] == True:
            img = Image.open(image).convert('RGB')
            img = transformation(img)
            val.append(img)
        else:
            img_orig = Image.open(image).convert('RGB')
            img = transformation(img_orig)
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

    return train, val, test

def collate_participant_eeg(idx_val, eeg_norm = False, apply_mean = False, all_idx = False, participant = None, data_path ='eeg_dataset/preprocessed', sub = '/sub-02'):

        if participant:
            train_file = np.load(participant + '/preprocessed_eeg_training.npy', allow_pickle = True).item()
            test_file = np.load(participant + '/preprocessed_eeg_test.npy', allow_pickle = True).item()
        else:
            train_file = np.load(data_path + sub + '/preprocessed_eeg_training.npy', allow_pickle = True).item()
            test_file = np.load(data_path + sub + '/preprocessed_eeg_test.npy', allow_pickle = True).item()

        # Train and val
        signal_data = train_file['preprocessed_eeg_data']
        chnames = train_file['ch_names']
        times = train_file['times']
        signal_data_val = signal_data[idx_val]
        signal_data = np.delete(signal_data, idx_val, 0)

        # Test
        signal_data_test = test_file['preprocessed_eeg_data']

        if apply_mean:
            signal_data = np.mean(signal_data, 1)
            signal_data_val = np.mean(signal_data_val, 1)
            signal_data_test = np.mean(signal_data_test, 1)

#         mean = np.mean(signal_data, axis=(0,1), keepdims=True)
#         std = np.std(signal_data, axis=(0,1), keepdims=True)
#         signal_data = (signal_data - mean) / std

#         mean = np.mean(signal_data_val, axis=(0,1), keepdims=True)
#         std = np.std(signal_data_val, axis=(0,1), keepdims=True)
#         signal_data_val = (signal_data_val - mean) / std

#         mean = np.mean(signal_data_test, axis=(0,1), keepdims=True)
#         std = np.std(signal_data_test, axis=(0,1), keepdims=True)
#         signal_data_test = (signal_data_test - mean) / std

        signal_data = torch.tensor(np.float32(signal_data))
        signal_data_val = torch.tensor(np.float32(signal_data_val))
        signal_data_test = torch.tensor(np.float32(signal_data_test))

        return signal_data, signal_data_val, signal_data_test, chnames, times



class EEGImagePairs(Dataset):
    def __init__(self, X, y, eeg_norm = True, transform = None, apply_mean = True, all_participants = False, test = False):

        self.X = X
        self.y = y
        self.transform = transform
        self.test = test
        self.mean_applied = apply_mean
        self.all_participants = all_participants
        self.eeg_norm = eeg_norm

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        eeg, _, _ = self.custom_min_max_scaler(self.X[idx])

        image_idx = idx

        if not self.mean_applied:
            image_idx = image_idx//4

        image = self.y[image_idx]

        return eeg, image

    def custom_min_max_scaler(self, x, a = 0, b = 1):
        min_val = torch.min(x)
        max_val = torch.max(x)
        x_scaled = a + (b - a) * (x - min_val) / (max_val - min_val)
        return x_scaled, min_val, max_val
        
        
def create_dataloaders(g_cpu, X_train, X_val, X_test, y_train, y_val, y_test, eeg_norm = True, apply_mean = True, all_participants = False, batch_size = 32):

    train_ds = EEGImagePairs(X_train, y_train, eeg_norm = eeg_norm, apply_mean= apply_mean, all_participants = all_participants)
    val_ds = EEGImagePairs(X_val, y_val, eeg_norm = eeg_norm, apply_mean= apply_mean)
    test_ds = EEGImagePairs(X_test, y_test, eeg_norm = eeg_norm, test = True)

    ### Convert the Datasets to PyTorch's Dataloader format ###
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=test_ds.__len__(), shuffle=False)

    if len(train_dl) > 0:
        print ('Loaded successfully! ')

    return train_dl, val_dl, test_dl

def get_dataloaders(g_cpu = torch.Generator(), eeg_norm = True, apply_mean = True, resolution = (64, 64), all_participants = False, batch_size = 32):
    idx_val = validation_strat()

    if os.path.isfile('dataloaders/train_dl.pt'):
        train_dl, val_dl, test_dl = torch.load('dataloaders/train_dl.pt'), torch.load('dataloaders/val_dl.pt'), torch.load('dataloaders/test_dl.pt')
        return train_dl, val_dl, test_dl
    
    elif os.path.isfile('dataloaders/eeg_train_data.pt'):
        X_train, X_val, X_test, _, _ = collate_participant_eeg(idx_val, apply_mean = apply_mean, to_torch = True)
    else:
        X_train, X_val, X_test, _, _ = collate_participant_eeg(idx_val, apply_mean = apply_mean, to_torch = True)
        if not apply_mean: 
            X_train = align_eegs(X_train)
            X_val = align_eegs(X_val)
            X_test = align_eegs(X_test)

    y_train, y_val, y_test = collate_images_dataset(idx_val, resolution = resolution)
    
    train_dl, val_dl, test_dl = create_dataloaders(g_cpu, X_train, X_val, X_test, y_train, y_val, y_test, eeg_norm = eeg_norm, apply_mean = apply_mean, all_participants = all_participants, batch_size = batch_size)

    torch.save(train_dl, 'dataloaders/train_dl.pt')
    torch.save(val_dl, 'dataloaders/val_dl.pt')
    torch.save(test_dl, 'dataloaders/test_dl.pt')

    return train_dl, val_dl, test_dl