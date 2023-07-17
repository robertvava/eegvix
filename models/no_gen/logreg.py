import torch 
import torch.nn as nn 


class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv1d(17, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64), 
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),  # Batch Normalization layer after ReLU
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(128 * 25, 512),  
            nn.ReLU(),
            nn.BatchNorm1d(512),  # Batch Normalization layer after ReLU

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),  # Batch Normalization layer after ReLU

            nn.Linear(256, 3 * 244 * 244),
            nn.Sigmoid()
        
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        x = x.view(-1, 3, 244, 244)
        return x
