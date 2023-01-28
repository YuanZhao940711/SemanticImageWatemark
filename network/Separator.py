import torch
import torch.nn as nn



class Separator(nn.Module):
    def __init__(self, latent_dim) :
        super(Separator, self).__init__()

        self.separator = nn.Sequential(
            nn.BatchNorm1d(num_features=latent_dim, affine=True),

            nn.Linear(in_features=latent_dim, out_features=latent_dim*2, bias=True),
            #nn.BatchNorm1d(num_features=latent_dim*2, affine=True),
            nn.LeakyReLU(inplace=True),

            nn.Linear(in_features=latent_dim*2, out_features=latent_dim*4, bias=True),
            #nn.BatchNorm1d(num_features=latent_dim*4, affine=True),
            nn.LeakyReLU(inplace=True),
        )

        self.output_layer = nn.Sequential(
            nn.Linear(in_features=latent_dim*4, out_features=latent_dim, bias=True),
            #nn.Tanh(),
        )


    def forward(self, feature_fused):

        feature_separated = self.separator(feature_fused)

        feature_separated = self.output_layer(feature_separated)

        return feature_separated