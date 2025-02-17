import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from stylegan2.model import EqualLinear
from encoders.helpers import get_blocks, Flatten, bottleneck_IR, bottleneck_IR_SE



class VanillaEncoder(nn.Module):
    def __init__(self, in_channels, latent_dim, device, hidden_dims=[16, 32, 64, 128, 256, 512, 1024, 2048]): # hidden_dims=[32, 64, 128, 256, 512, 1024, 2048]/[32, 64, 64, 128, 128, 256, 256, 512]
        super(VanillaEncoder, self).__init__()
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
            nn.Linear(in_features=hidden_dims[-1], out_features=hidden_dims[-1]),
            #nn.Tanh(),
        )
        self.fc_mu = nn.Linear(in_features=hidden_dims[-1], out_features=latent_dim)
        self.fc_logvar= nn.Linear(in_features=hidden_dims[-1], out_features=latent_dim)
    
    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick to sample N(mu, var) from N(0,1).
        """
        std = torch.exp(0.5 * log_var)

        eps = torch.rand_like(std)
        #eps = torch.normal(mean=0, std=1, size=std.shape).to(self.device)

        return eps * std + mu

    def forward(self, input):
        result = self.encoder(input) # bs*256*256*3 -> bs*1024*2*2

        result = self.output_layer(result) # bs*1024*2*2 -> bs*2048
        
        mu = self.fc_mu(result)
        log_var = self.fc_logvar(result)

        latent_z = self.reparameterize(mu, log_var)

        return mu, log_var, latent_z



class PspEncoder(nn.Module):
    def __init__(self, num_layers, mode='ir'):
        super(PspEncoder, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE

        self.input_layer = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64),
            )

        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = nn.Sequential(*modules)

        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(num_features=512),
            nn.AdaptiveAvgPool2d((7,7)),
            nn.Flatten(),
            nn.Linear(in_features=512 * 7 * 7, out_features=512),
        )

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x) # bsx512x16x16
        x = self.output_layer(x) # bsx512
        return x