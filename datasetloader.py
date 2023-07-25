import numpy as np 
import os 
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import torch
from sklearn.utils import resample
import glob



class DataSetLoader:

    @classmethod
    def collate_images_dataset(self, idx_val, resolution = (244, 244), reduce = False, limit_samples = False, data_dir = '/eeg_dataset'):
        transformation = transforms.Compose([
                    transforms.Resize((resolution[0],resolution[1])),
                    transforms.ToTensor(), 
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
                    ])
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
        
    
    @classmethod
    def collate_participant_eeg(self, idx_val, apply_mean = False, reduce = False, limit_samples = False, all_idx = False, participant = None, data_path ='eeg_dataset/preprocessed', sub = '/sub-02', to_torch = False):

        if participant:
            train_file = np.load(participant + '/preprocessed_eeg_training.npy', allow_pickle = True).item()
            test_file = np.load(participant + '/preprocessed_eeg_test.npy', allow_pickle = True).item()
        else:
            train_file = np.load(data_path + sub + '/preprocessed_eeg_training.npy', allow_pickle = True).item()
            test_file = np.load(data_path + sub + '/preprocessed_eeg_test.npy', allow_pickle = True).item()


        

        # Train and val
        signal_data_test = test_file['preprocessed_eeg_data']
        signal_data = train_file['preprocessed_eeg_data']
        chnames = train_file['ch_names']
        times = train_file['times']
     
        if apply_mean: 
            signal_data = np.mean(signal_data, 1)
            signal_data_test = np.mean(signal_data, 1)
            
        if reduce: 
            signal_data_reduced = signal_data[limit_samples]
            signal_data_val = signal_data[idx_val]
            signal_data = np.delete(signal_data_reduced, all_idx, 0)
            
        else: 
            signal_data_val = signal_data[idx_val]
            signal_data = np.delete(signal_data, idx_val, 0)
        

        # Test
        
        

        if to_torch:
            signal_data = torch.tensor(np.float32(signal_data))
            signal_data_val = torch.tensor(np.float32(signal_data_val))
            signal_data_test = torch.tensor(np.float32(signal_data))
            if reduce: 
                signal_data_reduced = torch.tensor(np.float32(signal_data_reduced))
            
        if reduce: 
            return signal_data_reduced, signal_data_val, signal_data_test, chnames, times
        return signal_data, signal_data_val, signal_data_test, chnames, times

    @classmethod
    def align_eegs(self, X, apply_mean = False):
        X =  torch.swapaxes(X, 0, 1)

        if len(X) > 200:
            if apply_mean:
                X = X.reshape(-1, 17, 100)
            else: 
                X = X.reshape(-1, 4, 17, 100)
                X = X.reshape(-1, 17, 100)
        else:
            if apply_mean:
                X = X.reshape(-1, 17, 100)
            else: 
                X = X.reshape(-1, 80, 17, 100)
                X = X.reshape(-1, 17, 100)

        return X

    @classmethod 
    def get_validation_strat(self, n_samples = 100):

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

    @classmethod 
    def get_limit_samples(self, idx_val, n_samples = 180):
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

    @classmethod 
    def collate_all_eeg_data(self, idx_val, apply_mean = False):
    
        subs_paths = DataSetLoader.paths_to_subjects()
        subs_train_data = []
        subs_test_data = []
        subs_val_data = []
        
        for sub in subs_paths: 
            signal_data, signal_data_val, signal_data_test, _, _ = DataSetLoader.collate_participant_eeg(idx_val, apply_mean = apply_mean, participant = sub)
            subs_train_data.append(signal_data)
            subs_val_data.append(signal_data_val)
            subs_test_data.append(signal_data_test)
        return torch.tensor(np.float32(subs_train_data)), torch.tensor(np.float32(subs_val_data)), torch.tensor(np.float32(subs_test_data))

    @classmethod 
    def get_all_idx(self, idx_val, limit_samples):
        
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

    @classmethod 
    def paths_to_npy(self, path, test_data = False):
        """
        :return: EEG Test paths, Training paths.
        """
        if test_data:
            return glob.glob(path + "/sub-**/*_test.npy", recursive=True)

        return glob.glob(path + "/sub-**/*_training.npy", recursive=True)

    @classmethod 
    def paths_to_subjects(self, path = os.getcwd() + '/eeg_dataset' + '/preprocessed'):
        return glob.glob(path + "/sub-**", recursive = True)

    @classmethod
    def load_data(self, all_participants = False, apply_mean = False, reduced_dataset = False, align_eegs = False):


        idx_val = DataSetLoader.get_validation_strat()

        if reduced_dataset: 

            reduced_samples = DataSetLoader.get_limit_samples(idx_val, n_samples = 180)
            y_train, y_val, y_test = DataSetLoader.collate_images_dataset(idx_val, reduce=True, limit_samples=reduced_samples)

            if all_participants:
                X_train, X_val, X_test = DataSetLoader.collate_all_eeg_data(idx_val, apply_mean = apply_mean, reduce = True, limit_samples = reduced_samples)
            else: 
                X_train, X_val, X_test, _, _ = DataSetLoader.collate_participant_eeg(idx_val, apply_mean = apply_mean, reduce = True, limit_samples = reduced_samples, to_torch = True)

            return X_train, X_val, X_test, y_train, y_val, y_test

        y_train, y_val, y_test = DataSetLoader.collate_images_dataset(idx_val)

        if all_participants:
            X_train, X_val, X_test = DataSetLoader.collate_all_eeg_data(idx_val, apply_mean = apply_mean)
        else: 
            X_train, X_val, X_test, _, _ = DataSetLoader.collate_participant_eeg(idx_val, apply_mean = apply_mean, to_torch = True)

        if align_eegs: 
            X_train = DataSetLoader.align_eegs(X_train, apply_mean = apply_mean)
            X_val = DataSetLoader.align_eegs(X_val, apply_mean = apply_mean)
            X_test = DataSetLoader.align_eegs(X_test, apply_mean = apply_mean)
        
        return X_train, X_val, X_test, y_train, y_val, y_test



# dls = DataSetLoader.load_data(all_participants = True, apply_mean = True)