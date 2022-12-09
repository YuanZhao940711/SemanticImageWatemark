import torch
import torch.nn as nn



class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dims=[32, 64, 128, 256, 512, 1024, 2048]):
        
        super(Decoder, self).__init__()

        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1])

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

            nn.ConvTranspose2d(in_channels=hidden_dims[-1], out_channels=3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1),
            nn.Tanh()
        )


    def forward(self, latent_z):
        result = self.decoder_input(latent_z) # bs*2048 -> bs*2048

        result = result.view(-1, 2048, 1, 1) # bs*2048*1*1

        result = self.decoder(result) # bs*32*64*64

        result = self.final_layer(result) # bs*3*256*256

        return result
