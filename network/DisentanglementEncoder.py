import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import namedtuple



class SEModule(nn.Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(
            channels, channels // reduction, kernel_size=1, padding=0 ,bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(
            channels // reduction, channels, kernel_size=1, padding=0 ,bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x

    
def conv4x4(in_c, out_c, norm=nn.InstanceNorm2d):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=4, stride=2, padding=1),
        norm(num_features=out_c, affine=True),
        nn.LeakyReLU(0.1, inplace=True)
    )


class deconv4x4(nn.Module):
    def __init__(self, in_c, out_c, norm=nn.InstanceNorm2d):
        super(deconv4x4, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels=in_c, out_channels=out_c, kernel_size=4, stride=2, padding=1)
        self.norm = norm(out_c, affine=True)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, input, feat):
        x = self.deconv(input)
        x = self.norm(x)
        x = self.lrelu(x)
        return torch.cat((x, feat), dim=1)


class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, stride, norm):
        super(ResidualBlock, self).__init__()
        if in_c == out_c:
            self.shortcut_layer = nn.MaxPool2d(1, stride)
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=1, stride=stride, bias=False),
                #nn.InstanceNorm2d(num_features=out_c, affine=True)
                norm(num_features=out_c, affine=True)
            )
        
        self.res_layer = nn.Sequential(
            #nn.InstanceNorm2d(num_features=in_c, affine=True),
            norm(num_features=in_c, affine=True),
            nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PReLU(num_parameters=out_c),
            nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=3, stride=stride, padding=1, bias=False),
            #nn.InstanceNorm2d(num_features=in_c, affine=True),
            norm(num_features=out_c, affine=True),
            #SEModule(channels=out_c, reduction=16)
        )

    def forward(self, input):
        shortcut = self.shortcut_layer(input)
        res = self.res_layer(input)

        return res + shortcut


class Bottleneck(namedtuple('Block', ['in_c', 'out_c', 'stride', 'norm'])):
    '''A named tuple describing a ResNet block.'''

def get_block(unit_module, in_channel, out_channel, num_units, stride=2, norm=nn.InstanceNorm2d):
    modules = []
    blocks = [Bottleneck(in_channel, out_channel, stride, norm)] + [Bottleneck(out_channel, out_channel, 1, norm) for i in range(num_units-1)]
    
    for block in blocks:
        modules.append(
            unit_module(
                block.in_c,
                block.out_c,
                block.stride,
                block.norm,
            )
        )
    body = nn.Sequential(*modules)

    return body


class DisentanglementEncoder(nn.Module):
    def __init__(self, latent_dim, norm=nn.InstanceNorm2d):
        super(DisentanglementEncoder, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(num_features=32),
            nn.PReLU(num_parameters=32)
        ) # bsx3x256x256 -> bsx32x256x256

        self.downsample_blocks_1 = get_block(unit_module=ResidualBlock, in_channel=32, out_channel=32, num_units=2, norm=norm) # bsx32x256x256 -> bsx32x128x128
        self.downsample_blocks_2 = get_block(unit_module=ResidualBlock, in_channel=32, out_channel=64, num_units=2, norm=norm) # bsx32x128x128 -> bsx64x64x64
        self.downsample_blocks_3 = get_block(unit_module=ResidualBlock, in_channel=64, out_channel=128, num_units=2, norm=norm) # bsx64x64x64 -> bsx128x32x32
        self.downsample_blocks_4 = get_block(unit_module=ResidualBlock, in_channel=128, out_channel=256, num_units=2, norm=norm) # bsx128x32x32 -> bsx256x16x16
        self.downsample_blocks_5 = get_block(unit_module=ResidualBlock, in_channel=256, out_channel=512, num_units=2, norm=norm) # bsx256x16x16 -> bsx512x8x8
        self.downsample_blocks_6 = get_block(unit_module=ResidualBlock, in_channel=512, out_channel=1024, num_units=2, norm=norm) # bsx512x8x8 -> bsx1024x4x4
        self.downsample_blocks_7 = get_block(unit_module=ResidualBlock, in_channel=1024, out_channel=2048, num_units=2, norm=norm) # bsx1024x4x4 -> bsx2048x2x2

        self.id_encoder = nn.Sequential(
            #nn.InstanceNorm2d(num_features=1024, affine=True),
            nn.Dropout(0.4),
            nn.Flatten(),
            nn.Linear(2048*2*2, latent_dim),
            #nn.InstanceNorm1d(num_features=latent_dim, affine=True),
            nn.Tanh()
        )

        self.upsample_block_0 = nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.upsample_block_1 = deconv4x4(in_c=1024, out_c=1024)
        self.upsample_block_2 = deconv4x4(in_c=2048, out_c=512)
        self.upsample_block_3 = deconv4x4(in_c=1024, out_c=256)
        self.upsample_block_4 = deconv4x4(in_c=512, out_c=128)
        self.upsample_block_5 = deconv4x4(in_c=256, out_c=64)
        self.upsample_block_6 = deconv4x4(in_c=128, out_c=32)

    def forward(self, input):
        feat_input = self.input_layer(input) # input: bsx3x256x256 -> feat_input: bsx32x256x256

        feat1 = self.downsample_blocks_1(feat_input) # feat1: bsx32x128x128
        feat2 = self.downsample_blocks_2(feat1) # feat2: bsx64x64x64
        feat3 = self.downsample_blocks_3(feat2) # feat3: bsx128x32x32
        feat4 = self.downsample_blocks_4(feat3) # feat4: bsx256x16x16
        feat5 = self.downsample_blocks_5(feat4) # feat5: bsx512x8x8
        feat6 = self.downsample_blocks_6(feat5) # feat6: bsx1024x4x4
        feat7 = self.downsample_blocks_7(feat6) # feat7: bsx2048x2x2

        id_feat = self.id_encoder(feat7) # feat7: bsx2048x2x2 -> id_feat: bsx512

        z_att1 = self.upsample_block_0(feat7) # feat7: bsx2048x2x2 -> z_att1: bsx1024x2x2
        z_att2 = self.upsample_block_1(z_att1, feat6) # z_att2: 2048 x 4 x 4
        z_att3 = self.upsample_block_2(z_att2, feat5) # z_att3: 1024 x 8 x 8
        z_att4 = self.upsample_block_3(z_att3, feat4) # z_att4: 512 x 16 x 16
        z_att5 = self.upsample_block_4(z_att4, feat3) # z_att5: 256 x 32 x 32
        z_att6 = self.upsample_block_5(z_att5, feat2) # z_att6: 128 x 64 x 64
        z_att7 = self.upsample_block_6(z_att6, feat1) # z_att7: 64 x 128 x 128
        z_att8 = F.interpolate(z_att7, scale_factor=2, mode='bilinear', align_corners=True) # z_att7: 64 x 256 x 256

        att_feat = [z_att1, z_att2, z_att3, z_att4, z_att5, z_att6, z_att7, z_att8]

        return id_feat, att_feat