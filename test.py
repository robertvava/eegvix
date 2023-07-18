import torch
import os 
import dataloaders.load_dl
from collate_utils import collate_images_dataset, collate_participant_eeg
from misc_utils import get_validation_strat
import dill as pickle


g_cpu = torch.Generator()
idx_val = get_validation_strat()
X_train, X_val, X_test, chnames, times = collate_participant_eeg(idx_val, to_torch = True)
y_train, y_val, y_test = collate_images_dataset(idx_val)
train, test, val = dataloaders.load_dl.create_dataloaders(g_cpu, X_train, X_val, X_test, y_train, y_val, y_test)

torch.save(train, 'dataloaders/train_dl.pth')
train_dl = torch.load('dataloaders/train_dl.pth')
diff = torch.load('models/diff/stable_diffusion/v1-5-pruned.ckpt')
print (train_dl)
print ("DIff model: ", diff)