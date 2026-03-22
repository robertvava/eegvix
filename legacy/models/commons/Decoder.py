import torch
import torch.nn as nn 

class Decoder(nn.Module):
    def __init__(self, out_channels):
        super(Decoder, self).__init__()
