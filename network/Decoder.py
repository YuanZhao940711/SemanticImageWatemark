import torch
import torch.nn as nn

from typing import List
from torch import tensor as Tensor


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dims=[32, 64, 128, 256, 512]):
        
        super(Decoder, self).__init__()

        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels=hidden_dims[i], out_channels=hidden_dims[i+1], kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i+1]),
                    nn.LeakyReLU()
                )
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels=hidden_dims[-1], out_channels=hidden_dims[-1], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=hidden_dims[-1], out_channels=3, kernel_size=3, padding=1),
            nn.Tanh()
        )


    def forward(self, latent_z: Tensor) -> Tensor:
        result = self.decoder_input(latent_z)
        result = result.view(-1, 512, 2, 2)

        result = self.decoder(result)

        result = self.final_layer(result)

        return result
