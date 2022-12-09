import torch
import torch.nn as nn



class Separator(nn.Module):
    def __init__(self, latent_dim) :
        super(Separator, self).__init__()

        self.separator = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=latent_dim*2),
            nn.LeakyReLU(),
            nn.Linear(in_features=latent_dim*2, out_features=latent_dim*2),
            nn.LeakyReLU(),
            nn.Linear(in_features=latent_dim*2, out_features=latent_dim),
            nn.Sigmoid()
        )


    def forward(self, feature_fused):

        feature_separated = self.separator(feature_fused)

        return feature_separated