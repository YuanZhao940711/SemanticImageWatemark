import torch
import torch.nn as nn



class Encoder(nn.Module):
    def __init__(self, in_channels, latent_dim, hidden_dims=[16, 32, 64, 128, 256, 512, 1024, 2048]): # hidden_dims=[32, 64, 128, 256, 512, 1024, 2048]/[32, 64, 64, 128, 128, 256, 256, 512]
        super(Encoder, self).__init__()

        modules = []

        for h_dim in hidden_dims[:-1]:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=in_channels, out_channels=h_dim, kernel_size=3, stride=2, padding=1),
                    #nn.BatchNorm2d(h_dim),
                    nn.InstanceNorm2d(h_dim, affine=True),
                    nn.LeakyReLU(inplace=True)
                )
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules) # bs*3*256*256 -> bs*1024*2*2

        self.output_layer = nn.Sequential(
            nn.Conv2d(in_channels=hidden_dims[-2], out_channels=hidden_dims[-1], kernel_size=3, stride=2, padding=1), # bs*1024*2*2 -> bs*2048*1*1
            nn.Flatten(), # bs*2048*1*1 -> bs*2048
            nn.Linear(in_features=hidden_dims[-1], out_features=latent_dim), # bs*2048 -> bs*512
            #nn.BatchNorm1d(latent_dim, affine=True)
            nn.Tanh()
        )

    def forward(self, input):
        result = self.encoder(input) # bs*256*256*3 -> bs*1024*2*2

        latent_z = self.output_layer(result) # bs*1024*2*2 -> bs*512

        return latent_z