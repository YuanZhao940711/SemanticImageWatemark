import torch
import torch.nn as nn

from typing import List
from torch import tensor as Tensor


class Fuser(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super(Fuser, self).__init__()

        self.fuser = nn.Sequential(
            nn.Linear(in_features=latent_dim*2, out_features=latent_dim*4),
            nn.LeakyReLU(),
            nn.Linear(in_features=latent_dim*4, out_features=latent_dim*2),
            nn.LeakyReLU(),
            nn.Linear(in_features=latent_dim*2, out_features=latent_dim),
            nn.Sigmoid()
        )
        

    def forward(self, feature_vector_a: Tensor, feature_vector_b: Tensor) -> Tensor:

        feature_vectors = torch.cat((feature_vector_a, feature_vector_b), dim=0)

        feature_fused = self.adopter(feature_vectors)

        return feature_fused