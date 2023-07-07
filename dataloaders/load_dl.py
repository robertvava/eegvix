from torch.utils.data import DataLoader, Dataset

def create_dataloaders(g_cpu, X_train, X_val, X_test, y_train, y_val, y_test):

    class EegDataset(Dataset):
        def __init__(self, X, y, transformation = None, target_transform = None):
            self.X = X
            self.y = y

        def __len__(self):
            return len(self.y)

        def __getitem__(self, idx):
            image = self.X[idx]
            target = self.y[idx]

            return image, target
        
        def __printself_(self):
            print (self.X)
            print (self.y)

    train_ds = EegDataset(X_train, y_train)
    val_ds = EegDataset(X_val, y_val)
    test_ds = EegDataset(X_test, y_test)

    ### Convert the Datasets to PyTorch's Dataloader format ###
    train_dl = DataLoader(train_ds, batch_size=16, shuffle=True,
        generator=g_cpu)
    val_dl = DataLoader(val_ds, batch_size=val_ds.__len__(), shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=test_ds.__len__(), shuffle=False)

    if len(train_dl) > 0:
        print ('Loaded successfully! ')

    return train_dl, val_dl, test_dl