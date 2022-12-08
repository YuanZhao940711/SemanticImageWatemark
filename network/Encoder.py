import torch
import torch.nn as nn

from typing import List
from torch import tensor as Tensor



class Encoder(nn.Module):
    def __init__(self, in_channels: int, latent_dim: int, hidden_dims=[32, 64, 128, 256, 512]) -> None:

        super(Encoder, self).__init__()

        modules = []

        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=in_channels, out_channels=h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU()
                )
            )
            in_channels = h_dim
        
        self.encoder = nn.Sequential(*modules)

        self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)


    def forward(self, input: Tensor) -> Tensor:
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        latent_z = eps * std + mu

        return latent_z
