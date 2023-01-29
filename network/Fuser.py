import torch
import torch.nn as nn



class Fuser(nn.Module):
    def __init__(self, latent_dim):
        super(Fuser, self).__init__()

        self.cover_input = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=latent_dim*2, bias=True),
            nn.BatchNorm1d(num_features=latent_dim*2, affine=True),
            nn.LeakyReLU(inplace=True),

            nn.Linear(in_features=latent_dim*2, out_features=latent_dim, bias=True),
            nn.LeakyReLU(inplace=True),
        )

        self.secret_input = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=latent_dim*2, bias=True),
            nn.BatchNorm1d(num_features=latent_dim*2, affine=True),
            nn.LeakyReLU(inplace=True),

            nn.Linear(in_features=latent_dim*2, out_features=latent_dim, bias=True),
            nn.LeakyReLU(inplace=True),
        )

        self.fuse_layer = nn.Sequential(             
            nn.Linear(in_features=latent_dim*2, out_features=latent_dim*4, bias=True),
            nn.BatchNorm1d(num_features=latent_dim*4, affine=True),
            nn.LeakyReLU(inplace=True),

            nn.Linear(in_features=latent_dim*4, out_features=latent_dim*4, bias=True),
            #nn.BatchNorm1d(num_features=latent_dim*4, affine=True),
            nn.LeakyReLU(inplace=True),

            nn.Linear(in_features=latent_dim*4, out_features=latent_dim*2, bias=True),
            #nn.BatchNorm1d(num_features=latent_dim*4, affine=True),
            nn.LeakyReLU(inplace=True),
        )
        
        self.output_layer = nn.Sequential(
            nn.Linear(in_features=latent_dim*2, out_features=latent_dim, bias=True),
            #nn.Tanh(),
        )
        

    def forward(self, cover_id, secret_feat):
        cover_vector = self.cover_input(cover_id)
        secret_vector = self.cover_input(secret_feat)

        feature_vectors = torch.cat((cover_vector, secret_vector), dim=1)

        feature_fused = self.fuse_layer(feature_vectors)

        return self.output_layer(feature_fused)