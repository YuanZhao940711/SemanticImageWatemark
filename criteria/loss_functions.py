import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from criteria.lpips.lpips import LPIPS



class GANLoss(nn.Module):
    def __init__(self, gan_mode='hinge', target_real_label=1.0, target_fake_label=0.0, tensor=torch.FloatTensor, opt=None):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.zero_tensor = None
        self.Tensor = tensor
        self.gan_mode = gan_mode
        self.opt = opt

        if gan_mode == 'ls':
            pass
        elif gan_mode == 'original':
            pass
        elif gan_mode == 'w':
            pass
        elif gan_mode == 'hinge':
            pass
        else:
            raise ValueError('Unexpected gan_mode {}'.format(gan_mode))
    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            return torch.ones_like(input).detach()
        else:
            return torch.zeros_like(input).detach()
    def get_zero_tensor(self, input):
        return torch.zeros_like(input).detach()
    def loss(self, input, target_is_real):
        # CrossEntropy Loss
        if self.gan_mode == 'original':
            target_tensor = self.get_target_tensor(input, target_is_real)
            loss = F.binary_cross_entropy_with_logits(input, target_tensor)
            return loss
        # Least Square Loss
        elif self.gan_mode == 'ls':
            target_tensor = self.get_target_tensor(input, target_is_real)
            return F.mse_loss(input, target_tensor)
        # Hinge Loss
        elif self.gan_mode == 'hinge':
            if target_is_real:
                loss = torch.mean(torch.relu(1 - input))
            else:
                loss = torch.mean(torch.relu(1 + input))
            return loss
        # Wgan Loss
        else:
            if target_is_real:
                return -input.mean()
            else:
                return input.mean()
    def __call__(self, input, target_is_real):
        if isinstance(input, list):
            loss = 0
            for pred_i in input:
                if isinstance(pred_i, list):
                    pred_i = pred_i[-1]
                loss_tensor = self.loss(pred_i, target_is_real)
                bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
                new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
                loss += new_loss
            return loss / len(input)
        else:
            return self.loss(input, target_is_real)



class AttLoss(nn.Module):
    def __init__(self):
        super(AttLoss, self).__init__()
        self.criterion = nn.MSELoss()

    def forward(self, att_input, att_output):
        assert len(att_input) == len(att_output), 'attributes maps are not equal'
        att_loss = 0
        for i in range(len(att_input)):
            att_loss += self.criterion(att_input[i], att_output[i])
        att_loss *= 0.5
        return att_loss



class IdLoss(nn.Module):
    def __init__(self, loss_mode):
        super(IdLoss, self).__init__()

        if loss_mode == 'MSE':
            self.criterion = nn.MSELoss()
            self.mode = loss_mode
        elif loss_mode == 'MAE':
            self.criterion = nn.L1Loss()
            self.mode = loss_mode
        elif loss_mode == 'Cos':
            self.mode = loss_mode
        else:
            raise ValueError('Unexpected Loss Mode {}'.format(loss_mode))

    def forward(self, id_input, id_output):
        if self.mode == 'MSE':
            id_loss = self.criterion(id_input, id_output)
            #id_loss *= 0.5
        elif self.mode == 'MAE':
            id_loss = self.criterion(id_input, id_output)
        elif self.mode == 'Cos':
            id_loss = torch.mean(1 - torch.cosine_similarity(id_input, id_output, dim=1))
        return id_loss



class RecConLoss(nn.Module):
    def __init__(self, loss_mode, device):
        super(RecConLoss, self).__init__()

        self.mode = loss_mode
        if self.mode == 'l2':
            self.criterion = nn.MSELoss().to(device)
        elif self.mode == 'lpips':
            self.criterion = LPIPS(net_type='alex').to(device).eval()
        else:
            raise ValueError('Unexpected Loss Mode {}'.format(loss_mode))            
    def forward(self, img_input, img_output):
        if self.mode == 'l2':
            rec_loss = 0.5 * self.criterion(img_input, img_output)
        elif self.mode == 'lpips':
            rec_loss = self.criterion(img_input, img_output)
        return rec_loss



class RecSecLoss(nn.Module):
    def __init__(self, loss_mode, device):
        super(RecSecLoss, self).__init__()
        self.mode = loss_mode
        if self.mode == 'l2':
            self.criterion = nn.MSELoss().to(device)
        elif self.mode == 'lpips':
            self.criterion = LPIPS(net_type='alex').to(device).eval()
        else:
            raise ValueError('Unexpected Loss Mode {}'.format(loss_mode))

    def forward(self, img_input, img_output):
        if self.mode == 'l2':
            rec_loss = 0.5 * self.criterion(img_input, img_output)
        elif self.mode == 'lpips':
            rec_loss = self.criterion(img_input, img_output)
        return rec_loss



class FeatLoss(nn.Module):
    def __init__(self, loss_mode):
        super(FeatLoss, self).__init__()

        if loss_mode == 'MSE':
            self.criterion = nn.MSELoss()
            self.mode = loss_mode
        elif loss_mode == 'MAE':
            self.criterion = nn.L1Loss()
            self.mode = loss_mode
        elif loss_mode == 'Cos':
            self.mode = loss_mode
        else:
            raise ValueError('Unexpected Loss Mode {}'.format(loss_mode))

    def forward(self, feat_rec, feat_ori):
        if self.mode == 'MSE':
            feat_loss = self.criterion(feat_rec, feat_ori)
        elif self.mode == 'MAE':
            feat_loss = self.criterion(feat_rec, feat_ori)
        elif self.mode == 'Cos':
            feat_loss = torch.mean(1 - torch.cosine_similarity(feat_rec, feat_ori, dim=1))
        
        return feat_loss



class KlLoss(nn.Module):
    def __init__(self):
        super(KlLoss, self).__init__()
        
    def forward(self, mu, log_var):
        kl_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim=0)
        return kl_loss