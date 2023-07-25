import torch
import os 
from dataloaders.load_dl import create_dataloaders
from datasetloader import DataSetLoader

import dill as pickle


g_cpu = torch.Generator()
X_train, X_val, X_test, y_train, y_val, y_test = DataSetLoader.load_data()
train, test, val = create_dataloaders(g_cpu, X_train, X_val, X_test, y_train, y_val, y_test)

torch.save(train, 'dataloaders/train_dl.pt')
train_dl = torch.load('dataloaders/train_dl.pt')
diff = torch.load('models/diff/stable_diffusion/v1-5-pruned.ckpt')
print ("DIff model: ", diff)