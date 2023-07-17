import torch
import torch.nn as nn
from torch.nn import functional as F

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv1d(17, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc_mu = nn.Linear(128*50, latent_dim)
        self.fc_var = nn.Linear(128*50, latent_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 128 * 50)
        self.deconv1 = nn.ConvTranspose1d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc_out = nn.Linear(64*100, 3*244*244)

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), 128, 50)
        x = F.relu(self.bn1(self.deconv1(x)))
        x = x.view(x.size(0), -1)
        x_hat = torch.sigmoid(self.fc_out(x)).view(x.size(0), 3, 244, 244)
        return x_hat

class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar


class VAELoss(nn.Module):
    def __init__(self):
        super(VAELoss, self).__init__()
        self.gaussian_nll = nn.GaussianNLLLoss(reduction='sum')

    @staticmethod
    def kl_divergence(mu, logvar):
        # KL divergence between q(z|x) and p(z), where p(z)~N(0,I)
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    def reconstruction_loss(self, x, x_hat):
        # assuming x_hat is the mean of a distribution with fixed standard deviation
        return self.gaussian_nll(x_hat, x, torch.ones_like(x_hat))

    def forward(self, x, x_hat, mu, logvar):
        return self.kl_divergence(mu, logvar) + self.reconstruction_loss(x, x_hat)
