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



class GradualStyleBlock(nn.Module):
    def __init__(self, in_c, out_c, spatial):
        super(GradualStyleBlock, self).__init__()
        self.out_c = out_c
        self.spatial = spatial
        num_pools = int(np.log2(spatial))
        modules = []
        modules += [nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU()]
        for i in range(num_pools - 1):
            modules += [
                nn.Conv2d(out_c, out_c, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU()
            ]
        self.convs = nn.Sequential(*modules)
        self.linear = EqualLinear(out_c, out_c, lr_mul=1)

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self.out_c)
        x = self.linear(x)
        return x



class GradualStyleEncoder(nn.Module):
    def __init__(self, num_layers, mode='ir', opts=None):
        super(GradualStyleEncoder, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = nn.Sequential(
            nn.Conv2d(opts.input_nc, 64, (3, 3), 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = nn.Sequential(*modules)

        self.styles = nn.ModuleList()
        self.style_count = opts.n_styles
        self.coarse_ind = 3
        self.middle_ind = 7
        for i in range(self.style_count):
            if i < self.coarse_ind:
                style = GradualStyleBlock(512, 512, 16)
            elif i < self.middle_ind:
                style = GradualStyleBlock(512, 512, 32)
            else:
                style = GradualStyleBlock(512, 512, 64)
            self.styles.append(style)
        self.latlayer1 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y

    def forward(self, x):
        x = self.input_layer(x)

        latents = []
        modulelist = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 6:
                c1 = x
            elif i == 20:
                c2 = x
            elif i == 23:
                c3 = x

        for j in range(self.coarse_ind):
            latents.append(self.styles[j](c3))

        p2 = self._upsample_add(c3, self.latlayer1(c2))
        for j in range(self.coarse_ind, self.middle_ind):
            latents.append(self.styles[j](p2))

        p1 = self._upsample_add(p2, self.latlayer2(c1))
        for j in range(self.middle_ind, self.style_count):
            latents.append(self.styles[j](p1))

        out = torch.stack(latents, dim=1)
        return out



class BackboneEncoderUsingLastLayerIntoW(nn.Module):
    def __init__(self, num_layers, mode='ir', opts=None):
        super(BackboneEncoderUsingLastLayerIntoW, self).__init__()
        #print('[*]Using BackboneEncoderUsingLastLayerIntoW')
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = nn.Sequential(
            #nn.Conv2d(opts.input_nc, 64, (3, 3), 1, 1, bias=False),
            nn.Conv2d(3, 64, (3, 3), 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64))
        self.output_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = EqualLinear(512, 512, lr_mul=1)
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = nn.Sequential(*modules)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_pool(x)
        x = x.view(-1, 512)
        x = self.linear(x)
        return x



class BackboneEncoderUsingLastLayerIntoWPlus(nn.Module):
    def __init__(self, num_layers, mode='ir'):
        super(BackboneEncoderUsingLastLayerIntoWPlus, self).__init__()
        print('Using BackboneEncoderUsingLastLayerIntoWPlus')
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.n_styles = 18 #opts.n_styles
        self.input_layer = nn.Sequential(
            #nn.Conv2d(opts.input_nc, 64, (3, 3), 1, 1, bias=False),
            nn.Conv2d(3, 64, (3, 3), 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64),
            )
        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 512),
            )
        self.linear = EqualLinear(512, 512 * self.n_styles, lr_mul=1)
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = nn.Sequential(*modules)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer(x)
        x = self.linear(x)
        x = x.view(-1, self.n_styles, 512)
        return x
