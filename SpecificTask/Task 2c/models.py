"""
models.py - Dataset and Model Definitions

This module contains the dataset classes for unlabelled and labelled jet images,
as well as the VAE and Classifier model architectures used in the project.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import h5py
from torch.utils.data import Dataset

# Dataset Definitions
class UnlabelledJetDataset(Dataset):
    """Dataset class for unlabelled jet images stored in an HDF5 file."""
    def __init__(self, h5_file, dataset_name='jet'):
        self.h5_file = h5_file
        self.dataset_name = dataset_name
        with h5py.File(h5_file, 'r') as f:
            self._len = len(f[dataset_name])

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            jet_image = f[self.dataset_name][idx]  # Shape: [125, 125, 8]
        jet_image = np.transpose(jet_image, (2, 0, 1))  # [8, 125, 125]
        for c in range(jet_image.shape[0]):
            channel_max = np.max(jet_image[c])
            if channel_max > 0:
                jet_image[c] /= channel_max
        jet_image = np.pad(jet_image, pad_width=((0,0),(1,2),(1,2)), mode='constant', constant_values=0)
        return torch.tensor(jet_image, dtype=torch.float32)

class LabelledJetDataset(Dataset):
    """Dataset class for labelled jet images stored in an HDF5 file."""
    def __init__(self, h5_file, dataset_name='jet', label_name='Y'):
        self.h5_file = h5_file
        self.dataset_name = dataset_name
        self.label_name = label_name
        with h5py.File(h5_file, 'r') as f:
            self._len = len(f[dataset_name])

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            jet_image = f[self.dataset_name][idx]  # [125, 125, 8]
            label = f[self.label_name][idx].item()
        jet_image = np.transpose(jet_image, (2, 0, 1))  # [8, 125, 125]
        for c in range(jet_image.shape[0]):
            channel_max = np.max(jet_image[c])
            if channel_max > 0:
                jet_image[c] /= channel_max
        jet_image = np.pad(jet_image, pad_width=((0,0),(1,2),(1,2)), mode='constant', constant_values=0)
        return torch.tensor(jet_image, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

# VAE Architecture
class VAE(nn.Module):
    """Variational Autoencoder for encoding and reconstructing jet images."""
    def __init__(self, input_shape=(8, 128, 128), latent_dim=128):
        super(VAE, self).__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Flatten()
        )
        self.fc_enc = nn.Sequential(
            nn.Linear(128*8*8, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, 128*8*8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def encode(self, x):
        h = self.encoder(x)
        h = self.fc_enc(h)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        logvar = torch.clamp(logvar, min=-10, max=10)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def decode(self, z):
        h = F.leaky_relu(self.decoder_input(z), 0.2)
        h = h.view(-1, 128, 8, 8)
        return self.decoder(h)

# Classifier Definition
class Classifier(nn.Module):
    """Classifier that uses the VAE's latent space for classification."""
    def __init__(self, vae, latent_dim=128):
        super(Classifier, self).__init__()
        self.vae = vae
        self.classifier_net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        mu, _ = self.vae.encode(x)
        return self.classifier_net(mu)

# Loss Function
def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """Loss function for VAE training, combining reconstruction and KL divergence."""
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss

if __name__ == "__main__":
    print("This is models.py. It contains dataset and model definitions for the project.")