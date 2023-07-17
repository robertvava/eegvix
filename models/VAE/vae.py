import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

    def forward(self, x):
        z, mu, log_var = 1
        return z, mu, log_var

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()