import torch
import torch.nn as nn 

class Encoder(nn.Module):
    def __init__(self, out_channels):
        super(Encoder, self).__init__()
