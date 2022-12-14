import torch
import torch.nn as nn



class Encoder(nn.Module):
    def __init__(self, in_channels, latent_dim, hidden_dims=[16, 32, 64, 128, 256, 512, 1024, 2048]): # hidden_dims=[32, 64, 128, 256, 512, 1024, 2048]/[32, 64, 64, 128, 128, 256, 256, 512]
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

        self.output_layer = nn.Sequential(
            nn.Flatten(), # bs*2048*1*1 -> bs*2048
            nn.Linear(in_features=hidden_dims[-1], out_features=latent_dim), # bs*2048 -> bs*512
            nn.BatchNorm1d(latent_dim, affine=True)
        )
        #self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
        #self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)
        #self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        #self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)

    def forward(self, input):
        result = self.encoder(input) # bs*256*256*3 -> bs*2048*1*1
        """
        result = torch.flatten(result, start_dim=1) # bs*512
        
        mu = self.fc_mu(result) # bs*512 -> bs*512
        log_var = self.fc_var(result) # bs*512 -> bs*512

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        latent_z = eps * std + mu # bs*512
        """
        latent_z = self.output_layer(result) # bs*2048*1*1 -> bs*512

        #return latent_z, mu, log_var
        return latent_z
