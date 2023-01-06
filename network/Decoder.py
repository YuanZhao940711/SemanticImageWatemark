import torch
import torch.nn as nn



class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dims=[16, 32, 64, 128, 256, 512, 1024, 2048]): # hidden_dims=[32, 64, 128, 256, 512, 1024, 2048]/[32, 64, 64, 128, 128, 256, 256, 512]
        super(Decoder, self).__init__()

        modules = []

        self.input_linear = nn.Linear(latent_dim, hidden_dims[-1])
        
        hidden_dims.reverse()
        
        self.input_deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=hidden_dims[0], out_channels=hidden_dims[1], kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.LeakyReLU(inplace=True),
        )

        for i in range(len(hidden_dims) - 2):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels=hidden_dims[i+1], out_channels=hidden_dims[i+2], kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
                    nn.InstanceNorm2d(hidden_dims[i+2], affine=True),
                    nn.LeakyReLU(inplace=True),
                )
            )

        self.decoder = nn.Sequential(*modules)

        self.output_layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels=hidden_dims[-1], out_channels=3, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.InstanceNorm2d(num_features=3, affine=True),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(num_features=3, affine=True),
            #nn.LeakyReLU(inplace=True),

            #nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0),
            nn.Tanh(),
        )


    def forward(self, latent_z):
        result = self.input_linear(latent_z) # bs*512 -> bs*2048
        result = result.view(result.shape[0], result.shape[-1], 1, 1) # bs*2048*1*1

        result = self.input_deconv(result) # bs*2048*1*1 -> bs*1024*2*2

        result = self.decoder(result) # bs*1024*2*2 -> bs*16*128*128

        result = self.output_layer(result) # bs*16*128*128 -> bs*3*256*256

        return result

