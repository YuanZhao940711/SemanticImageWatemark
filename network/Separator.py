import torch
import torch.nn as nn



class Separator(nn.Module):
    def __init__(self, latent_dim) :
        super(Separator, self).__init__()

        self.separator = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=latent_dim*2),
            nn.BatchNorm1d(num_features=latent_dim*2, affine=True),
            nn.LeakyReLU(inplace=True),

            nn.Linear(in_features=latent_dim*2, out_features=latent_dim*2),
            nn.BatchNorm1d(num_features=latent_dim*2, affine=True),
            nn.LeakyReLU(inplace=True),
        )

        self.output_layer = nn.Sequential(
            nn.Linear(in_features=latent_dim*2, out_features=latent_dim),
            nn.BatchNorm1d(num_features=latent_dim, affine=True),
            nn.Tanh()
        )


    def forward(self, feature_fused):

        feature_separated = self.separator(feature_fused)

        feature_separated = self.output_layer(feature_separated)

        return feature_separated