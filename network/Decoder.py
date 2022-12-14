import torch
import torch.nn as nn



class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dims=[16, 32, 64, 128, 256, 512, 1024, 2048]): # hidden_dims=[32, 64, 128, 256, 512, 1024, 2048]/[32, 64, 64, 128, 128, 256, 256, 512]
        super(Decoder, self).__init__()

        modules = []

        self.input_layer = nn.Linear(latent_dim, hidden_dims[-1])
        #self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1]*4)

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

        self.output_layer = nn.Sequential(
            #nn.ConvTranspose2d(in_channels=hidden_dims[-1], out_channels=hidden_dims[-1], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ConvTranspose2d(in_channels=hidden_dims[-1], out_channels=3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),

            #nn.ConvTranspose2d(in_channels=hidden_dims[-1], out_channels=3, kernel_size=3, stride=2, padding=1, output_padding=1),
            #nn.BatchNorm2d(3),
            #nn.LeakyReLU(),

            #nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1),
            
            #nn.Conv2d(in_channels=hidden_dims[-1], out_channels=3, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )


    def forward(self, latent_z):
        result = self.input_layer(latent_z) # bs*512 -> bs*2048
        result = result.view(result.shape[0], result.shape[-1], 1, 1) # bs*2048*1*1

        result = self.decoder(result) # bs*2048*1*1 -> bs*16*128*128

        result = self.output_layer(result) # bs*16*128*128 -> bs*3*256*256

        return result

