import torch
import torch.nn as nn



class Fuser(nn.Module):
    def __init__(self, latent_dim):
        super(Fuser, self).__init__()

        self.fuser = nn.Sequential(
            nn.Linear(in_features=latent_dim*2, out_features=latent_dim*4, bias=False),
            nn.BatchNorm1d(num_features=latent_dim*4, affine=True),
            nn.LeakyReLU(inplace=True),

            nn.Linear(in_features=latent_dim*4, out_features=latent_dim*2, bias=False),
            nn.BatchNorm1d(num_features=latent_dim*2, affine=True),
            nn.LeakyReLU(inplace=True),
        )
        self.output_layer = nn.Sequential(
            nn.Linear(in_features=latent_dim*2, out_features=latent_dim, bias=False),
            nn.Tanh()
        )
        

    def forward(self, feature_vector_a, feature_vector_b):

        feature_vectors = torch.cat((feature_vector_a, feature_vector_b), dim=1)

        feature_fused = self.fuser(feature_vectors)

        feature_fused = self.output_layer(feature_fused)

        return feature_fused