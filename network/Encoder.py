import torch
import torch.nn as nn



class Encoder(nn.Module):
    def __init__(self, in_channels, latent_dim, device, hidden_dims=[16, 32, 64, 128, 256, 512, 1024, 2048]): # hidden_dims=[32, 64, 128, 256, 512, 1024, 2048]/[32, 64, 64, 128, 128, 256, 256, 512]
        super(Encoder, self).__init__()
        self.device = device

        modules = []

        for h_dim in hidden_dims[:-1]:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=in_channels, out_channels=h_dim, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.InstanceNorm2d(h_dim, affine=True),
                    nn.LeakyReLU(inplace=True),
                )
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules) # bs*3*256*256 -> bs*1024*2*2

        self.output_layer = nn.Sequential(
            nn.Conv2d(in_channels=hidden_dims[-2], out_channels=hidden_dims[-1], kernel_size=3, stride=2, padding=1, bias=False), # bs*1024*2*2 -> bs*2048*1*1
            nn.LeakyReLU(inplace=True),
            nn.Flatten(), # bs*2048*1*1 -> bs*2048
        )

        self.fc_mu = nn.Linear(in_features=latent_dim*4, out_features=latent_dim)
        self.fc_logsigma2 = nn.Linear(in_features=latent_dim*4, out_features=latent_dim)
    
    def reparameterize(self, mu, log_sigma_2):
        """
        Reparameterization trick to sample N(mu, var) from N(0,1).
        """
        std = torch.exp(0.5 * log_sigma_2)

        #eps = torch.rand_like(std)
        eps = torch.normal(mean=0, std=1, size=std.shape).to(self.device)

        return eps * std + mu

    def forward(self, input):
        result = self.encoder(input) # bs*256*256*3 -> bs*1024*2*2

        result = self.output_layer(result) # bs*1024*2*2 -> bs*2048

        mu = self.fc_mu(result)
        log_sigma_2 = self.fc_logsigma2(result)

        latent_z = self.reparameterize(mu, log_sigma_2)

        return latent_z, mu, log_sigma_2