from torch.utils.data import DataLoader, Dataset
import torch

def create_dataloaders(g_cpu, X_train, X_val, X_test, y_train, y_val, y_test):


    X_train = torch.swapaxes(X_train, 0, 1)
    X_train = X_train.reshape(-1, 4, 17, 100)
    X_train = X_train.reshape(-1, 17, 100)


    class EegDataset(Dataset):
        def __init__(self, X, y):
            self.X = X
            self.y = y

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            target = self.X[idx]
            # If condition is met, then the repetitions were not collapsed into a mean. 
            if len(self.X) > 170000:
                image = self.y[idx//(10*4)] 
            else:
                image = self.y[idx//10] 
        
            return target, image
    
        def __printself_(self):
            print (self.X)
            print (self.y)

    train_ds = EegDataset(X_train, y_train)
    val_ds = EegDataset(X_val, y_val)
    test_ds = EegDataset(X_test, y_test)

    ### Convert the Datasets to PyTorch's Dataloader format ###
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=False,
        generator=g_cpu)
    val_dl = DataLoader(val_ds, batch_size=val_ds.__len__(), shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=test_ds.__len__(), shuffle=False)

    if len(train_dl) > 0:
        print ('Loaded successfully! ')

    try:
        len(train_dl) > 0
        print ("Loaded successfully! \n")
    except: 
        print ("Failed to load")

    return train_dl, val_dl, test_dl