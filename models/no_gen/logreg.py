import torch.nn as nn

class RegressionModel(nn.Module):
    def __init__(self, resolution = 64):
        
        super(RegressionModel, self).__init__()
        self.resolution = resolution
        
        self.conv1 =  nn.Sequential(
            nn.Conv1d(17, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64), 
            nn.MaxPool1d(kernel_size=2, stride=2)
            )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),  # Batch Normalization layer after ReLU
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
    
        self.fc1 = nn.Sequential(nn.Linear(128 * 25, 512),  
            nn.ReLU(),
            nn.BatchNorm1d(512)  # Batch Normalization layer after ReLU
        )
        self.fc2 =nn.Sequential(nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),  # Batch Normalization layer after ReLU
        )
        
        self.fc3 = nn.Sequential(
            nn.Linear(256, 3 * self.resolution * self.resolution),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = x.view(-1, 3, self.resolution, self.resolution)
        return x
