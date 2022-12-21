import torch
import torch.nn as nn
import torch.nn.functional as F



class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm2d(out_c, affine=True)
        self.conv2 = nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=3, stride=1, padding=dilation, dilation=dilation, groups=group, bias=False)
        self.norm2 = nn.InstanceNorm2d(out_c, affine=True)

    def forward(self, x):
        y = F.relu(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))

        return F.relu(x+y)



def conv4x4(c_in, c_out, norm=nn.BatchNorm2d):
    return nn.Sequential(
        nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=4, stride=2, padding=1),
        norm(c_out),
        nn.LeakyReLU(0.1, inplace=True)
    )



class deconv4x4(nn.Module):
    def __init__(self, c_in, c_out, norm=nn.BatchNorm2d):
        super(deconv4x4, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels=c_in, out_channels=c_out, kernel_size=4, stride=2, padding=1)
        self.bn = norm(c_out)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, input, feat):
        x = self.deconv(input)
        x = self.bn(x)
        x = self.lrelu(x)
        return torch.cat((x, feat), dim=1)



class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)



class DisentanglementEncoder(nn.Module):
    def __init__(self):
        super(DisentanglementEncoder, self).__init__()

        self.conv1 = conv4x4(3, 32)
        self.conv2 = conv4x4(32, 64)
        self.conv3 = conv4x4(64, 128)
        self.conv4 = conv4x4(128, 256)
        self.conv5 = conv4x4(256, 512)
        self.conv6 = conv4x4(512, 1024)
        self.conv7 = conv4x4(1024, 1024)

        self.id_encoder = nn.Sequential(
            nn.BatchNorm2d(1024),
            nn.Dropout(0.4),
            nn.Flatten(),
            nn.Linear(1024*4*4, 512),
            nn.BatchNorm1d(512, affine=True),
            nn.Tanh()
        )

        self.deconv1 = deconv4x4(1024, 1024)
        self.deconv2 = deconv4x4(2048, 512)
        self.deconv3 = deconv4x4(1024, 256)
        self.deconv4 = deconv4x4(512, 128)
        self.deconv5 = deconv4x4(256, 64)
        self.deconv6 = deconv4x4(128, 32)
    
    def forward(self, input):
        feat1 = self.conv1(input) # input: bsx3x256x256 -> feat1: bsx32x128x128
        feat2 = self.conv2(feat1) # feat2: bsx64x64x64
        feat3 = self.conv3(feat2) # feat3: bsx128x32x32
        feat4 = self.conv4(feat3) # feat4: bsx256x16x16
        feat5 = self.conv5(feat4) # feat5: bsx512x8x8
        feat6 = self.conv6(feat5) # feat6: bsx1024x4x4

        id_feat = self.id_encoder(feat6) # feat6: bsx1024x4x4 - bsx512

        z_att1 = self.conv7(feat6) # z_att1: 1024x2x2

        z_att2 = self.deconv1(z_att1, feat6) # z_att2: 2048 x 4 x 4
        z_att3 = self.deconv2(z_att2, feat5) # z_att3: 1024 x 8 x 8
        z_att4 = self.deconv3(z_att3, feat4) # z_att4: 512 x 16 x 16
        z_att5 = self.deconv4(z_att4, feat3) # z_att5: 256 x 32 x 32
        z_att6 = self.deconv5(z_att5, feat2) # z_att6: 128 x 64 x 64
        z_att7 = self.deconv6(z_att6, feat1) # z_att7: 64 x 128 x 128

        z_att8 = F.interpolate(z_att7, scale_factor=2, mode='bilinear', align_corners=True) # z_att7: 64 x 256 x 256
        
        att_feat = [z_att1, z_att2, z_att3, z_att4, z_att5, z_att6, z_att7, z_att8]

        return id_feat, att_feat