from datasetloader import DataSetLoader
from dataloaders.load_dl import *
import torch.optim as optim
import torch
import wandb
import torch.nn as nn
from models.no_gen.logreg import RegressionModel
import os

def run_pipeline(model_config, reduce = False):

    if model_config['act'] == 'train':

        g_cpu = torch.Generator()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        
        X_train, X_val, X_test, y_train, y_val, y_test = DataSetLoader.load_data()
        # if os.path.isfile('dataloaders/train_dl.pt'):
        #     train_dl = torch.load('dataloaders/train_dl.pt')
        #     val_dl = torch.load('dataloaders/val_dl.pt')
        #     test_dl = torch.load('dataloaders/test_dl.pt')
        # else: 
        train_dl, val_dl, test_dl = create_dataloaders(g_cpu, X_train, X_val, X_test, y_train, y_val, y_test)
            # torch.save(train_dl, 'dataloaders/train_dl.pt')
            # torch.save(val_dl, 'dataloaders/val_dl.pt')
            # torch.save(test_dl, 'dataloaders/test_dl.pt')

        train(train_dl, model_config, device)

    elif model_config['act'] == 'generate':
        return 1

def train(train_dl, config, device = 'cpu'):
    print ("Model name: ", config['model'])
    
