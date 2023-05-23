import os
import sys
import json
import time
import numpy as np

sys.path.append(".")
sys.path.append("..")

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from options.options import TrainSihnAlphaOptions

from face_modules.model import Backbone
from network.MAE import MLAttrEncoder
from network.AAD import AADGenerator
from network.Fuser import Fuser
from network.Separator import Separator
from network.Encoder import PspEncoder
from network.Decoder import Decoder # VanillaDecoder
#from stylegan2.model import Stylegan2Decoder

import piq
from criteria import loss_functions

from utils.dataset import ImageDataset
from utils.common import weights_init, alignment, visualize_results, print_log, log_metrics, l2_norm, AverageMeter




class Train:
    def __init__(self, args):
        self.args = args

        torch.backends.deterministic = True
        torch.backends.cudnn.benchmark = True
        SEED = self.args.seed
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

        self.args.device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
        print_log('[*]Running on device: {}'.format(self.args.device), self.args.logpath)


        ##### Initialize networks and load pretrained models #####
        # Id Encoder
        self.idencoder = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se').to(self.args.device)
        #"""
        try:
            #self.idencoder.load_state_dict(torch.load(os.path.join(self.args.checkpoint_dir, 'Id_best.pth'), map_location=self.args.device), strict=True)
            self.idencoder.load_state_dict(torch.load(os.path.join(self.args.checkpoint_dir, 'model_ir_se50.pth'), map_location=self.args.device), strict=True)
            print_log("[*]Successfully loaded Id Encoder's pre-trained model", self.args.logpath)
        except:
            raise ValueError("[*]Unable to load Id Encoder's pre-trained model")
            #print_log("[*]Training Id Encoder from scratch", self.args.logpath)
            #self.idencoder.apply(weights_init)
        #"""
        #print_log("[*]Training Id Encoder from scratch", self.args.logpath)
        #self.idencoder.apply(weights_init)
        
        # Att Encoder
        self.attencoder = MLAttrEncoder().to(self.args.device)
        #"""
        try:
            self.attencoder.load_state_dict(torch.load(os.path.join(self.args.checkpoint_dir, 'Att_best.pth'), map_location=self.args.device), strict=True)
            print_log("[*]Successfully loaded Att Encoder's pre-trained model", self.args.logpath)
        except:
            #raise ValueError("[*]Unable to load Att Encoder's pre-trained model")
            print_log("[*]Training Att Encoder from scratch", self.args.logpath)
            self.attencoder.apply(weights_init)
        #"""
        #print_log("[*]Training Disentangler AttEncoder from scratch", self.args.logpath)
        #self.attencoder.apply(weights_init)

        # AAD Generator
        self.generator= AADGenerator(c_id=512).to(self.args.device)
        #"""
        try:
            self.generator.load_state_dict(torch.load(os.path.join(self.args.checkpoint_dir, 'Gen_best.pth'), map_location=self.args.device), strict=True)
            print_log("[*]Successfully loaded AAD Generator's pre-trained model", self.args.logpath)
        except:
            #raise ValueError("[*]Unable to load Generator's pre-trained model")
            print_log("[*]Training AAD Generator from scratch", self.args.logpath)
            self.generator.apply(weights_init)
        #"""
        #print_log("[*]Training AAD Generator from scratch", self.args.logpath)
        #self.generator.apply(weights_init)
        
        # Encoder 
        self.encoder = PspEncoder(num_layers=50, mode='ir_se').to(self.args.device)
        #"""
        try:
            self.encoder.load_state_dict(torch.load(os.path.join(self.args.checkpoint_dir, 'Encoder_best.pth'), map_location=self.args.device), strict=True)
            print_log("[*]Successfully loaded Encoder's pre-trained model", self.args.logpath)
        except:
            #raise ValueError("[*]Unable to load Encoder's pre-trained model")
            print_log("[*]Training Encoder from scratch", self.args.logpath)
            self.encoder.apply(weights_init)
        #"""
        #print_log("[*]Training Encoder from scratch", self.args.logpath)
        #self.encoder.apply(weights_init)

        # Decoder 
        #default size=1024, style_dim=512, n_mlp=8
        #self.decoder = Stylegan2Decoder(size=self.args.image_size, style_dim=self.args.latent_dim, n_mlp=8).to(self.args.device)
        #self.decoder = VanillaDecoder(latent_dim=self.args.latent_dim).to(self.args.device)
        self.decoder = Decoder(latent_dim=self.args.latent_dim).to(self.args.device)
        #"""
        try:
            self.decoder.load_state_dict(torch.load(os.path.join(self.args.checkpoint_dir, 'Decoder_best.pth'), map_location=self.args.device), strict=True)
            print_log("[*]Successfully loaded Decoder's pre-trained model", self.args.logpath)
        except:
            #raise ValueError("[*]Unable to load Decoder's pre-trained model")
            print_log("[*]Training Decoder from scratch", self.args.logpath)
            self.decoder.apply(weights_init)
        #"""
        #print_log("[*]Training Decoder from scratch", self.args.logpath)
        #self.decoder.apply(weights_init)

        # Fuser 
        #"""
        self.fuser = Fuser(latent_dim=self.args.latent_dim).to(self.args.device)
        try:
            self.fuser.load_state_dict(torch.load(os.path.join(self.args.checkpoint_dir, 'Fuser_best.pth')))
            print_log("[*]Successfully loaded Fuser's pre-trained model", self.args.logpath)
        except:
            #raise ValueError("[*]Unable to load Fuser's pre-trained model")
            print_log("[*]Training Fuser from scratch", self.args.logpath)
            self.fuser.apply(weights_init)
        #"""

        # Separator 
        #"""
        self.separator = Separator(latent_dim=self.args.latent_dim).to(self.args.device)
        try:
            self.separator.load_state_dict(torch.load(os.path.join(self.args.checkpoint_dir, 'Separator_best.pth')))
            print_log("[*]Successfully loaded Separator's pre-trained model", self.args.logpath)
        except:
            #raise ValueError("[*]Unable to load Separator's pre-trained model")
            print_log("[*]Training Separator from scratch", self.args.logpath)
            self.separator.apply(weights_init)
        #"""


        ##### Initialize optimizers #####
        self.att_optim = optim.Adam(self.attencoder.parameters(), lr=self.args.att_lr, betas=(0.5, 0.999), weight_decay=0)
        self.gen_optim = optim.Adam(self.generator.parameters(), lr=self.args.gen_lr, betas=(0.5, 0.999), weight_decay=0)
        self.encoder_optim = optim.Adam(self.encoder.parameters(), lr=self.args.encoder_lr, betas=(0.5, 0.999), weight_decay=0)
        self.decoder_optim = optim.Adam(self.decoder.parameters(), lr=self.args.decoder_lr, betas=(0.5, 0.999), weight_decay=0)
        self.fuser_optim = optim.Adam(self.fuser.parameters(), lr=self.args.fuser_lr, betas=(0.5, 0.999), weight_decay=0)
        self.separator_optim = optim.Adam(self.separator.parameters(), lr=self.args.separator_lr, betas=(0.5, 0.999), weight_decay=0)

        #self.dis_scheduler = optim.lr_scheduler.ExponentialLR(optimizer=self.dis_optim, gamma=0.8, last_epoch=-1, verbose=True)
        #self.gen_scheduler = optim.lr_scheduler.ExponentialLR(optimizer=self.gen_optim, gamma=0.8, last_epoch=-1, verbose=True)
        #self.encoder_scheduler = optim.lr_scheduler.ExponentialLR(optimizer=self.encoder_optim, gamma=0.8, last_epoch=-1, verbose=True)
        #self.decoder_scheduler = optim.lr_scheduler.ExponentialLR(optimizer=self.decoder_optim, gamma=0.8, last_epoch=-1, verbose=True)
        #self.fuser_scheduler = optim.lr_scheduler.ExponentialLR(optimizer=self.fuser_optim, gamma=0.8, last_epoch=-1, verbose=True)
        #self.separator_scheduler = optim.lr_scheduler.ExponentialLR(optimizer=self.separator_optim, gamma=0.8, last_epoch=-1, verbose=True)

        """
        self.att_scheduler = optim.lr_scheduler.StepLR(optimizer=self.att_optim, step_size=self.args.step_size, gamma=0.8, last_epoch=-1,verbose=True)
        self.gen_scheduler = optim.lr_scheduler.StepLR(optimizer=self.gen_optim, step_size=self.args.step_size, gamma=0.8, last_epoch=-1,verbose=True)
        self.encoder_scheduler = optim.lr_scheduler.StepLR(optimizer=self.encoder_optim, step_size=self.args.step_size, gamma=0.8, last_epoch=-1,verbose=True)
        self.decoder_scheduler = optim.lr_scheduler.StepLR(optimizer=self.decoder_optim, step_size=self.args.step_size, gamma=0.8, last_epoch=-1, verbose=True)
        self.fuser_scheduler = optim.lr_scheduler.StepLR(optimizer=self.fuser_optim, step_size=self.args.step_size, gamma=0.8, last_epoch=-1,verbose=True)
        self.separator_scheduler = optim.lr_scheduler.StepLR(optimizer=self.separator_optim, step_size=self.args.step_size, gamma=0.8, last_epoch=-1, verbose=True)
        """

        ##### Initialize loss functions #####
        self.con_att_loss = loss_functions.AttLoss().to(self.args.device).eval()
        self.con_id_loss = loss_functions.IdLoss(self.args.conidloss_mode).to(self.args.device).eval()
        self.con_mse_loss = loss_functions.RecConLoss(self.args.conrecloss_mode, self.args.device).eval()
        self.con_lpips_loss = loss_functions.RecConLoss('lpips', self.args.device).eval()
        self.con_ssim_loss = loss_functions.RecConLoss('ssim', self.args.device).eval()
        
        self.sec_feat_loss = loss_functions.FeatLoss(self.args.secfeatloss_mode, self.args.device).eval()
        self.sec_mse_loss = loss_functions.RecSecLoss('l2', self.args.device).eval()
        self.sec_lpips_loss = loss_functions.RecSecLoss('lpips', self.args.device).eval()
        self.sec_ssim_loss = loss_functions.RecSecLoss('ssim', self.args.device).eval()


        ##### Initialize data loaders ##### 
        train_cover_transforms = transforms.Compose([
            transforms.Resize([self.args.image_size, self.args.image_size]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ])
        val_cover_transforms = transforms.Compose([
            transforms.Resize([self.args.image_size, self.args.image_size]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ])
        secret_transforms = transforms.Compose([
            transforms.Resize([self.args.image_size, self.args.image_size]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ])

        train_cover_dataset = ImageDataset(root=self.args.train_cover_dir, transforms=train_cover_transforms)
        val_cover_dataset = ImageDataset(root=self.args.val_cover_dir, transforms=val_cover_transforms)
        secret_dataset = ImageDataset(root=self.args.secret_dir, transforms=secret_transforms)
        print_log(info="[*]Loaded {} training cover images".format(len(train_cover_dataset)), log_path=self.args.logpath, console=True)
        print_log(info="[*]Loaded {} validation cover images".format(len(val_cover_dataset)), log_path=self.args.logpath, console=True)
        print_log(info="[*]Loaded {} secret images".format(len(secret_dataset)), log_path=self.args.logpath, console=True)

        self.train_cover_loader = DataLoader(
            train_cover_dataset,
            batch_size=self.args.train_bs,
            shuffle=True,
            num_workers=int(self.args.num_workers),
            drop_last=True
        )
        self.val_cover_loader = DataLoader(
            val_cover_dataset,
            batch_size=self.args.train_bs,
            shuffle=True,
            num_workers=int(self.args.num_workers),
            drop_last=True
        )
        self.secret_loader = DataLoader(
            secret_dataset,
            batch_size=self.args.secret_bs,
            shuffle=True,
            num_workers=int(self.args.num_workers),
            drop_last=True
        )


        ##### Initialize logger #####
        self.writer = SummaryWriter(log_dir=self.args.tensorboardlogs_dir)
        self.best_loss = None


    def forward_pass(self, cover, secret):
        cover = cover.to(self.args.device)

        cover_att = self.attencoder(cover)
        cover_id = self.idencoder(alignment(cover))
        cover_id_norm = l2_norm(cover_id) # 也许输入 fuser 的 cover_id 是不需要 norm的，可以等 fuser 合成好新的 feature 之后，再进行 norm

        secret = secret.to(self.args.device)

        #"""
        if secret.shape[0] == 1:
            secret_ori = secret.repeat(cover.shape[0]//2, 1, 1, 1)
            secret_null = torch.zeros(cover.shape[0] - secret_ori.shape[0], secret_ori.shape[1], secret_ori.shape[2], secret_ori.shape[3]).to(self.args.device)
            secret_input = torch.cat((secret_ori, secret_null), dim=0)
        else:
            secret_input = secret

        #"""
        secret_feature_ori = self.encoder(secret_ori)
        secret_feature_ori_norm = l2_norm(secret_feature_ori)
        secret_feature_null = torch.zeros(cover.shape[0] - secret_ori.shape[0], secret_feature_ori.shape[1]).to(self.args.device)

        #secret_feature_input = torch.cat((secret_feature_ori, secret_feature_null), dim=0)
        secret_feature_input = torch.cat((secret_feature_ori_norm, secret_feature_null), dim=0)
        #"""

        #input_feature = cover_id_norm + secret_feature_ori_norm
        fused_feature = self.fuser(cover_id=cover_id_norm[:cover.shape[0]//2], secret_feat=secret_feature_ori_norm)
        input_feature = torch.cat((fused_feature, cover_id_norm[cover.shape[0]//2:]), dim=0)
        input_feature_norm = l2_norm(input_feature)

        #container = self.generator(inputs=(cover_att, input_feature))
        container = self.generator(inputs=(cover_att, input_feature_norm))

        container_att = self.attencoder(container)
        container_id = self.idencoder(alignment(container))
        container_id_norm = l2_norm(container_id)

        #secret_feature_output = self.separator(container_id)
        secret_feature_output = self.separator(container_id_norm)
        secret_feature_output_norm = l2_norm(secret_feature_output)

        """
        secret_output, _ = self.decoder(
            styles=[secret_feature_output],
            #styles=[secret_feature_output_norm],
            #styles=[container_id_norm],
            input_is_latent=True,
            randomize_noise=True,
            return_latents=False,
        )
        """
        #secret_output = self.decoder(latent_z=secret_feature_output)
        secret_output = self.decoder(latent_z=secret_feature_output_norm)


        ##### Collect results ##### 
        data_dict = {
            'cover': cover,
            'container': container,
            'secret_ori': secret_input,
            'secret_output': secret_output,
            'input_feature': input_feature_norm,
            'cover_id': cover_id_norm,
            'container_id': container_id_norm,
            #'input_feature': input_feature,
            #'cover_id': cover_id,
            #'container_id': container_id,
            'secret_feature_input': secret_feature_input,
            #'secret_feature_output': secret_feature_output,
            #'secret_feature_input': secret_feature_input_norm,
            'secret_feature_output': secret_feature_output_norm,
            'cover_att': cover_att,
            'container_att': container_att,
        }

        return data_dict


    def training(self, epoch, cover_loader, secret_loader):
        self.attencoder.train()
        self.generator.train()
        self.encoder.train()
        self.decoder.train()
        self.fuser.train()
        self.separator.train()

        batch_time = AverageMeter()

        Con_Att_loss = AverageMeter()
        Con_Id_loss = AverageMeter()
        Con_Mse_loss = AverageMeter()
        Con_Lpips_loss = AverageMeter()
        Con_SSIM_loss = AverageMeter()
        Con_PSNR_loss = AverageMeter()

        Sec_Feat_loss = AverageMeter()
        Sec_Mse_loss = AverageMeter()
        Sec_Lpips_loss = AverageMeter()
        Sec_SSIM_loss = AverageMeter()
        Sec_PSNR_loss = AverageMeter()

        Train_losses = AverageMeter()

        Con_SSIM_index = AverageMeter()
        Sec_SSIM_index = AverageMeter()
        Con_PSNR_index = AverageMeter()
        Sec_PSNR_index = AverageMeter()

        start_time = time.time()

        secret_iterator = iter(secret_loader)
        for train_iter, cover_batch in enumerate(cover_loader):
            try:
                secret_batch = next(secret_iterator)
            except StopIteration:
                secret_iterator = iter(secret_loader)
                secret_batch = next(secret_iterator)


            ##### Training #####
            data_dict = self.forward_pass(cover_batch, secret_batch)

            loss_con_att = self.con_att_loss(data_dict['container_att'], data_dict['cover_att'])
            loss_con_id = self.con_id_loss(data_dict['container_id'], data_dict['input_feature'])
            #id_ratio = 0.5
            #loss_con_id = id_ratio*self.con_id_loss(data_dict['container_id'], data_dict['input_feature']) + (1-id_ratio)*self.con_id_loss(data_dict['container_id'], data_dict['cover_id'])
            loss_con_mse = self.con_mse_loss(data_dict['container'], data_dict['cover'])
            loss_con_lpips = self.con_lpips_loss(data_dict['container'], data_dict['cover'])
            loss_con_ssim = self.con_ssim_loss((data_dict['container']+1)/2, (data_dict['cover']+1)/2)
            #loss_con_ssim = self.con_ssim_loss(data_dict['container'], data_dict['cover'])
            loss_con_psnr = (torch.tensor(data=60, dtype=torch.float, device=self.args.device) - piq.psnr(x=(data_dict['container']+1)/2, y=(data_dict['cover']+1)/2, data_range=1., reduction='mean')) / torch.tensor(data=60, dtype=torch.float, device=self.args.device)
            
            loss_sec_feat = self.sec_feat_loss(data_dict['secret_feature_output'], data_dict['secret_feature_input'])
            loss_sec_mse = self.sec_mse_loss(data_dict['secret_output'], data_dict['secret_ori'])
            loss_sec_lpips = self.sec_lpips_loss(data_dict['secret_output'], data_dict['secret_ori'])
            loss_sec_ssim = self.sec_ssim_loss((data_dict['secret_output']+1)/2, (data_dict['secret_ori']+1)/2)
            #loss_sec_ssim = self.sec_ssim_loss(data_dict['secret_output'], data_dict['secret_ori'])
            loss_sec_psnr = (torch.tensor(data=60, dtype=torch.float, device=self.args.device) - piq.psnr(x=(data_dict['secret_output']+1)/2, y=(data_dict['secret_ori']+1)/2, data_range=1., reduction='mean')) / torch.tensor(data=60, dtype=torch.float, device=self.args.device)

            """
            Sum_train_losses = self.args.con_att_lambda*loss_con_att + self.args.con_id_lambda*loss_con_id + self.args.con_mse_lambda*loss_con_mse + self.args.con_lpips_lambda*loss_con_lpips \
            + self.args.sec_feat_lambda*loss_sec_feat + self.args.sec_mse_lambda*loss_sec_mse + self.args.sec_lpips_lambda*loss_sec_lpips
            """
            Sum_train_losses = self.args.con_att_lambda*loss_con_att + self.args.con_id_lambda*loss_con_id + self.args.con_mse_lambda*loss_con_mse + self.args.sec_feat_lambda*loss_sec_feat + self.args.sec_mse_lambda*loss_sec_mse
            #Sum_train_losses = self.args.con_att_lambda*loss_con_att + self.args.con_id_lambda*loss_con_id + self.args.con_mse_lambda*loss_con_mse + self.args.sec_feat_lambda*loss_sec_feat + self.args.sec_mse_lambda*loss_sec_mse + self.args.sec_lpips_lambda*loss_sec_lpips
            #Sum_train_losses = self.args.con_att_lambda*loss_con_att + self.args.con_id_lambda*loss_con_id + self.args.con_mse_lambda*loss_con_mse + self.args.con_lpips_lambda*loss_con_lpips + self.args.sec_feat_lambda*loss_sec_feat + self.args.sec_mse_lambda*loss_sec_mse + self.args.sec_lpips_lambda*loss_sec_lpips
            #Sum_train_losses = self.args.con_att_lambda*loss_con_att + self.args.con_id_lambda*loss_con_id + self.args.con_mse_lambda*loss_con_mse + self.args.con_lpips_lambda*loss_con_lpips + self.args.con_ssim_lambda*loss_con_ssim + self.args.sec_feat_lambda*loss_sec_feat + self.args.sec_mse_lambda*loss_sec_mse + self.args.sec_lpips_lambda*loss_sec_lpips + self.args.sec_ssim_lambda*loss_sec_ssim
            
            """
            Sum_train_losses = self.args.con_att_lambda*loss_con_att + self.args.con_id_lambda*loss_con_id + self.args.con_mse_lambda*loss_con_mse + self.args.con_lpips_lambda*loss_con_lpips + self.args.con_ssim_lambda*loss_con_ssim + self.args.con_psnr_lambda*loss_con_psnr \
                + self.args.sec_feat_lambda*loss_sec_feat + self.args.sec_mse_lambda*loss_sec_mse + self.args.sec_lpips_lambda*loss_sec_lpips + self.args.sec_ssim_lambda*loss_sec_ssim + self.args.sec_psnr_lambda*loss_sec_psnr
            """


            index_con_ssim = piq.ssim(x=(data_dict['container']+1)/2, y=(data_dict['cover']+1)/2, data_range=1., reduction='mean')
            index_sec_ssim = piq.ssim(x=(data_dict['secret_output']+1)/2, y=(data_dict['secret_ori']+1)/2, data_range=1., reduction='mean')

            index_con_psnr = piq.psnr(x=(data_dict['container']+1)/2, y=(data_dict['cover']+1)/2, data_range=1., reduction='mean')
            index_sec_psnr = piq.psnr(x=(data_dict['secret_output']+1)/2, y=(data_dict['secret_ori']+1)/2, data_range=1., reduction='mean')
            #index_con_psnr = piq.psnr(x=data_dict['container'], y=data_dict['cover'], data_range=1., reduction='mean')
            #index_sec_psnr = piq.psnr(x=data_dict['secret_output'], y=data_dict['secret_ori'], data_range=1., reduction='mean')

            self.att_optim.zero_grad()
            self.gen_optim.zero_grad()
            self.encoder_optim.zero_grad()
            self.decoder_optim.zero_grad()
            self.fuser_optim.zero_grad()
            self.separator_optim.zero_grad()

            Sum_train_losses.backward()

            self.att_optim.step()
            self.gen_optim.step()
            self.encoder_optim.step()
            self.decoder_optim.step()
            self.fuser_optim.step()
            self.separator_optim.step()


            ##### Log losses and computation time #####
            Con_Att_loss.update(loss_con_att.item(), self.args.train_bs)
            Con_Id_loss.update(loss_con_id.item(), self.args.train_bs)
            Con_Mse_loss.update(loss_con_mse.item(), self.args.train_bs)
            Con_Lpips_loss.update(loss_con_lpips.item(), self.args.train_bs)
            Con_SSIM_loss.update(loss_con_ssim.item(), self.args.train_bs)
            Con_PSNR_loss.update(loss_con_psnr.item(), self.args.train_bs)

            Sec_Feat_loss.update(loss_sec_feat.item(), self.args.train_bs)
            Sec_Mse_loss.update(loss_sec_mse.item(), self.args.train_bs)
            Sec_Lpips_loss.update(loss_sec_lpips.item(), self.args.train_bs)
            Sec_SSIM_loss.update(loss_sec_ssim.item(), self.args.train_bs)
            Sec_PSNR_loss.update(loss_sec_psnr.item(), self.args.train_bs)

            Train_losses.update(Sum_train_losses.item(), self.args.train_bs)

            Con_SSIM_index.update(index_con_ssim.item(), self.args.train_bs)
            Sec_SSIM_index.update(index_sec_ssim.item(), self.args.train_bs)
            Con_PSNR_index.update(index_con_psnr.item(), self.args.train_bs)
            Sec_PSNR_index.update(index_sec_psnr.item(), self.args.train_bs)

            batch_time.update(time.time()-start_time)
            start_time = time.time()

            train_data_dict = {
                'ConAttLoss': Con_Att_loss.avg,
                'ConIdLoss': Con_Id_loss.avg,
                'ConMseLoss': Con_Mse_loss.avg,
                'ConLpipsLoss': Con_Lpips_loss.avg,
                'ConSSIMLoss': Con_SSIM_loss.avg,
                'ConPSNRLoss': Con_PSNR_loss.avg,
                'SecFeatLoss': Sec_Feat_loss.avg,
                'SecMseLoss': Sec_Mse_loss.avg,
                'SecLpipsLoss': Sec_Lpips_loss.avg,
                'SecSSIMLoss': Sec_SSIM_loss.avg,
                'SecPSNRLoss': Sec_PSNR_loss.avg,
                'SumTrainLosses': Train_losses.avg,
                'ConSSIMIndex': Con_SSIM_index.avg,
                'SecSSIMIndex': Sec_SSIM_index.avg,
                'ConPSNRIndex': Con_PSNR_index.avg,
                'SecPSNRIndex': Sec_PSNR_index.avg,
            }


            ##### Board and log losses, and visualize results #####
            if (self.global_train_steps+1) % self.args.board_interval == 0:
                """
                train_log = "[{:d}/{:d}][Iteration: {:05d}][Steps: {:05d}] ConAtt_loss: {:.6f} ConId_loss: {:.6f} ConMse_loss: {:.6f} ConLpips_loss: {:.6f} SecFeat_loss: {:.6f} SecMse_loss: {:.6f} SecLpips_loss: {:.6f} Sumlosses={:.6f} BatchTime: {:.4f}".format(
                    epoch+1, self.args.max_epoch, train_iter+1, self.global_train_steps+1, Con_Att_loss.val, Con_Id_loss.val, Con_Mse_loss.val, Con_Lpips_loss.val, Sec_Feat_loss.val, Sec_Mse_loss.val, Sec_Lpips_loss.val, Train_losses.val, batch_time.val
                )
                """
                """
                train_log = "[{:d}/{:d}][Iteration: {:05d}][Steps: {:05d}] ConAtt_loss: {:.6f} ConId_loss: {:.6f} ConMse_loss: {:.6f} ConLpips_loss: {:.6f} ConSSIM_loss: {:.6f} ConPSNR_index: {:.6f} SecFeat_loss: {:.6f} SecMse_loss: {:.6f} SecLpips_loss: {:.6f} SecSSIM_loss: {:.6f} SecPSNR_index: {:.6f} Sumlosses={:.6f} BatchTime: {:.4f}".format(
                    epoch+1, self.args.max_epoch, train_iter+1, self.global_train_steps+1, Con_Att_loss.val, Con_Id_loss.val, Con_Mse_loss.val, Con_Lpips_loss.val, Con_SSIM_loss.val, Con_PSNR_index.val, Sec_Feat_loss.val, Sec_Mse_loss.val, Sec_Lpips_loss.val, Sec_SSIM_loss.val, Sec_PSNR_index.val, Train_losses.val, batch_time.val
                )
                """
                train_log = "[{:d}/{:d}][Iteration: {:05d}][Steps: {:05d}] ConAtt_loss: {:.6f} ConId_loss: {:.6f} ConMse_loss: {:.6f} ConLpips_loss: {:.6f} ConSSIM_loss: {:.6f} ConPSNR_loss: {:.6f} ConSSIM_index: {:.6f} ConPSNR_index: {:.6f} SecFeat_loss: {:.6f} SecMse_loss: {:.6f} SecLpips_loss: {:.6f} SecSSIM_loss: {:.6f} SecPSNR_loss: {:.6f} SecSSIM_index: {:.6f} SecPSNR_index: {:.6f} Sumlosses={:.6f} BatchTime: {:.4f}".format(
                    epoch+1, self.args.max_epoch, train_iter+1, self.global_train_steps+1, Con_Att_loss.val, Con_Id_loss.val, Con_Mse_loss.val, Con_Lpips_loss.val, Con_SSIM_loss.val, Con_PSNR_loss.val, Con_SSIM_index.val, Con_PSNR_index.val, Sec_Feat_loss.val, Sec_Mse_loss.val, Sec_Lpips_loss.val, Sec_SSIM_loss.val, Sec_PSNR_loss.val, Sec_SSIM_index.val, Sec_PSNR_index.val, Train_losses.val, batch_time.val
                )
                print_log(info=train_log, log_path=self.args.logpath, console=True)
                log_metrics(writer=self.writer, data_dict=train_data_dict, step=self.global_train_steps+1, prefix='train')

            if self.global_train_steps == 0 or (self.global_train_steps+1) % self.args.image_interval == 0:
                #visualize_sihn_results(vis_dict=data_dict, dis_num=self.args.display_num, epoch=epoch+1, prefix='train', save_dir=self.args.trainpics_dir, iter=train_iter+1, step=self.global_train_steps+1)
                visualize_results(vis_dict=data_dict, dis_num=self.args.display_num, epoch=epoch+1, prefix='train', save_dir=self.args.trainpics_dir, iter=train_iter+1, step=self.global_train_steps+1)
            
            self.global_train_steps += 1
            
            if train_iter == self.args.max_train_iters-1:
                break
        
        """
        train_epoch_log = "Training[{:d}/{:d}] ConAtt_loss: {:.6f} ConId_loss: {:.6f} ConMse_loss: {:.6f} ConLpips_loss: {:.6f} SecFeat_loss: {:.6f} SecMse_loss: {:.6f} SecLpips_loss: {:.6f} Sumlosses={:.6f} BatchTime: {:.4f}".format(
            epoch+1, self.args.max_epoch, Con_Att_loss.avg, Con_Id_loss.avg, Con_Mse_loss.avg, Con_Lpips_loss.avg, Sec_Feat_loss.avg, Sec_Mse_loss.avg, Sec_Lpips_loss.avg, Train_losses.avg, batch_time.sum
        )
        """
        """
        train_epoch_log = "Training[{:d}/{:d}] ConAtt_loss: {:.6f} ConId_loss: {:.6f} ConMse_loss: {:.6f} ConLpips_loss: {:.6f} ConSSIM_loss: {:.6f} ConPSNR_index: {:.6f} SecFeat_loss: {:.6f} SecMse_loss: {:.6f} SecLpips_loss: {:.6f} SecSSIM_loss: {:.6f} SecPSNR_index: {:.6f} Sumlosses={:.6f} BatchTime: {:.4f}".format(
            epoch+1, self.args.max_epoch, Con_Att_loss.avg, Con_Id_loss.avg, Con_Mse_loss.avg, Con_Lpips_loss.avg, Con_SSIM_loss.avg, Con_PSNR_index.avg, Sec_Feat_loss.avg, Sec_Mse_loss.avg, Sec_Lpips_loss.avg, Sec_SSIM_loss.avg, Sec_PSNR_index.avg, Train_losses.avg, batch_time.sum
        )
        """
        train_epoch_log = "Training[{:d}/{:d}] ConAtt_loss: {:.6f} ConId_loss: {:.6f} ConMse_loss: {:.6f} ConLpips_loss: {:.6f} ConSSIM_loss: {:.6f} ConPSNR_loss: {:.6f} ConSSIM_index: {:.6f} ConPSNR_index: {:.6f} SecFeat_loss: {:.6f} SecMse_loss: {:.6f} SecLpips_loss: {:.6f} SecSSIM_loss: {:.6f} SecPSNR_loss: {:.6f} SecSSIM_index: {:.6f} SecPSNR_index: {:.6f} Sumlosses={:.6f} BatchTime: {:.4f}".format(
            epoch+1, self.args.max_epoch, Con_Att_loss.avg, Con_Id_loss.avg, Con_Mse_loss.avg, Con_Lpips_loss.avg, Con_SSIM_loss.avg, Con_PSNR_loss.avg, Con_SSIM_index.avg, Con_PSNR_index.avg, Sec_Feat_loss.avg, Sec_Mse_loss.avg, Sec_Lpips_loss.avg, Sec_SSIM_loss.avg, Sec_PSNR_loss.avg, Sec_SSIM_index.avg, Sec_PSNR_index.avg, Train_losses.avg, batch_time.sum
        )
        print_log(info=train_epoch_log, log_path=self.args.logpath, console=True)



    def validation(self, epoch, cover_loader, secret_loader):
        self.attencoder.eval()
        self.generator.eval()
        self.encoder.eval()
        self.decoder.eval()
        self.fuser.eval()
        self.separator.eval()

        batch_time = AverageMeter()

        Con_Att_loss = AverageMeter()
        Con_Id_loss = AverageMeter()
        Con_Mse_loss = AverageMeter()
        Con_Lpips_loss = AverageMeter()
        Con_SSIM_loss = AverageMeter()
        Con_PSNR_loss = AverageMeter()

        Sec_Feat_loss = AverageMeter()
        Sec_Mse_loss = AverageMeter()
        Sec_Lpips_loss = AverageMeter()
        Sec_SSIM_loss = AverageMeter()
        Sec_PSNR_loss = AverageMeter()

        Val_losses = AverageMeter()

        Con_SSIM_index = AverageMeter()
        Sec_SSIM_index = AverageMeter()
        Con_PSNR_index = AverageMeter()
        Sec_PSNR_index = AverageMeter()

        start_time = time.time()

        secret_iterator = iter(secret_loader)
        for val_iter, cover_batch in enumerate(cover_loader):
            try:
                secret_batch = next(secret_iterator)
            except StopIteration:
                secret_iterator = iter(secret_loader)
                secret_batch = next(secret_iterator)

            data_dict = self.forward_pass(cover_batch, secret_batch)

            # Calculate losses
            loss_con_att = self.con_att_loss(data_dict['container_att'], data_dict['cover_att'])
            loss_con_id = self.con_id_loss(data_dict['container_id'], data_dict['input_feature'])
            #id_ratio = 0.5
            #loss_con_id = id_ratio*self.con_id_loss(data_dict['container_id'], data_dict['input_feature']) + (1-id_ratio)*self.con_id_loss(data_dict['container_id'], data_dict['cover_id'])
            loss_con_mse = self.con_mse_loss(data_dict['container'], data_dict['cover'])
            loss_con_lpips = self.con_lpips_loss(data_dict['container'], data_dict['cover'])
            loss_con_ssim = self.con_ssim_loss((data_dict['container']+1)/2, (data_dict['cover']+1)/2)
            #loss_con_ssim = self.con_ssim_loss(data_dict['container'], data_dict['cover'])
            loss_con_psnr = (torch.tensor(data=60, dtype=torch.float, device=self.args.device) - piq.psnr(x=(data_dict['container']+1)/2, y=(data_dict['cover']+1)/2, data_range=1., reduction='mean')) / torch.tensor(data=60, dtype=torch.float, device=self.args.device)
            
            loss_sec_feat = self.sec_feat_loss(data_dict['secret_feature_output'], data_dict['secret_feature_input'])
            loss_sec_mse = self.sec_mse_loss(data_dict['secret_output'], data_dict['secret_ori'])
            loss_sec_lpips = self.sec_lpips_loss(data_dict['secret_output'], data_dict['secret_ori'])
            loss_sec_ssim = self.sec_ssim_loss((data_dict['secret_output']+1)/2, (data_dict['secret_ori']+1)/2)
            #loss_sec_ssim = self.sec_ssim_loss(data_dict['secret_output'], data_dict['secret_ori'])
            loss_sec_psnr = (torch.tensor(data=60, dtype=torch.float, device=self.args.device) - piq.psnr(x=(data_dict['secret_output']+1)/2, y=(data_dict['secret_ori']+1)/2, data_range=1., reduction='mean')) / torch.tensor(data=60, dtype=torch.float, device=self.args.device)

            """
            sum_val_loss = self.args.con_att_lambda*loss_con_att + self.args.con_id_lambda*loss_con_id + self.args.con_mse_lambda*loss_con_mse + self.args.con_lpips_lambda*loss_con_lpips \
            + self.args.sec_feat_lambda*loss_sec_feat + self.args.sec_mse_lambda*loss_sec_mse + self.args.sec_lpips_lambda*loss_sec_lpips
            """
            sum_val_loss = self.args.con_att_lambda*loss_con_att + self.args.con_id_lambda*loss_con_id + self.args.con_mse_lambda*loss_con_mse + self.args.sec_feat_lambda*loss_sec_feat + self.args.sec_mse_lambda*loss_sec_mse
            #sum_val_loss = self.args.con_att_lambda*loss_con_att + self.args.con_id_lambda*loss_con_id + self.args.con_mse_lambda*loss_con_mse + self.args.sec_feat_lambda*loss_sec_feat + self.args.sec_mse_lambda*loss_sec_mse + self.args.sec_lpips_lambda*loss_sec_lpips
            #sum_val_loss = self.args.con_att_lambda*loss_con_att + self.args.con_id_lambda*loss_con_id + self.args.con_mse_lambda*loss_con_mse + self.args.con_lpips_lambda*loss_con_lpips + self.args.sec_feat_lambda*loss_sec_feat + self.args.sec_mse_lambda*loss_sec_mse + self.args.sec_lpips_lambda*loss_sec_lpips
            #sum_val_loss = self.args.con_att_lambda*loss_con_att + self.args.con_id_lambda*loss_con_id + self.args.con_mse_lambda*loss_con_mse + self.args.con_lpips_lambda*loss_con_lpips + self.args.con_ssim_lambda*loss_con_ssim + self.args.sec_feat_lambda*loss_sec_feat + self.args.sec_mse_lambda*loss_sec_mse + self.args.sec_lpips_lambda*loss_sec_lpips + self.args.sec_ssim_lambda*loss_sec_ssim
            
            """
            sum_val_loss = self.args.con_att_lambda*loss_con_att + self.args.con_id_lambda*loss_con_id + self.args.con_mse_lambda*loss_con_mse + self.args.con_lpips_lambda*loss_con_lpips + self.args.con_ssim_lambda*loss_con_ssim + self.args.con_psnr_lambda*loss_con_psnr \
                + self.args.sec_feat_lambda*loss_sec_feat + self.args.sec_mse_lambda*loss_sec_mse + self.args.sec_lpips_lambda*loss_sec_lpips + self.args.sec_ssim_lambda*loss_sec_ssim + self.args.sec_psnr_lambda*loss_sec_psnr
            """


            index_con_ssim = piq.ssim(x=(data_dict['container']+1)/2, y=(data_dict['cover']+1)/2, data_range=1., reduction='mean')
            index_sec_ssim = piq.ssim(x=(data_dict['secret_output']+1)/2, y=(data_dict['secret_ori']+1)/2, data_range=1., reduction='mean')

            index_con_psnr = piq.psnr(x=(data_dict['container']+1)/2, y=(data_dict['cover']+1)/2, data_range=1., reduction='mean')
            index_sec_psnr = piq.psnr(x=(data_dict['secret_output']+1)/2, y=(data_dict['secret_ori']+1)/2, data_range=1., reduction='mean')
            #index_con_psnr = piq.psnr(x=data_dict['container'], y=data_dict['cover'], data_range=1., reduction='mean')
            #index_sec_psnr = piq.psnr(x=data_dict['secret_output'], y=data_dict['secret_ori'], data_range=1., reduction='mean')

            Con_Att_loss.update(loss_con_att.item(), self.args.train_bs)
            Con_Id_loss.update(loss_con_id.item(), self.args.train_bs)
            Con_Mse_loss.update(loss_con_mse.item(), self.args.train_bs)
            Con_Lpips_loss.update(loss_con_lpips.item(), self.args.train_bs)
            Con_PSNR_loss.update(loss_con_psnr.item(), self.args.train_bs)

            Sec_Feat_loss.update(loss_sec_feat.item(), self.args.train_bs)
            Sec_Mse_loss.update(loss_sec_mse.item(), self.args.train_bs)
            Sec_Lpips_loss.update(loss_sec_lpips.item(), self.args.train_bs)
            Sec_SSIM_loss.update(loss_sec_ssim.item(), self.args.train_bs)
            Sec_PSNR_loss.update(loss_sec_psnr.item(), self.args.train_bs)

            Val_losses.update(sum_val_loss.item(), self.args.train_bs)

            Con_SSIM_index.update(index_con_ssim.item(), self.args.train_bs)
            Sec_SSIM_index.update(index_sec_ssim.item(), self.args.train_bs)
            Con_PSNR_index.update(index_con_psnr.item(), self.args.train_bs)
            Sec_PSNR_index.update(index_sec_psnr.item(), self.args.train_bs)

            batch_time.update(time.time() - start_time)
            start_time = time.time()

            if val_iter == self.args.max_val_iters-1:
                break

        validate_data_dict = {
            'ConAttLoss': Con_Att_loss.avg,
            'ConIdLoss': Con_Id_loss.avg,
            'ConMseLoss': Con_Mse_loss.avg,
            'ConLpipsLoss': Con_Lpips_loss.avg,
            'ConSSIMLoss': Con_SSIM_loss.avg,
            'ConPSNRLoss': Con_PSNR_loss.avg,
            'SecFeatLoss': Sec_Feat_loss.avg,
            'SecMseLoss': Sec_Mse_loss.avg,
            'SecLpipsLoss': Sec_Lpips_loss.avg,
            'SecSSIMLoss': Sec_SSIM_loss.avg,
            'SecPSNRLoss': Sec_PSNR_loss.avg,
            'SumValidateLosses': Val_losses.avg,
            'ConSSIMIndex': Con_SSIM_index.avg,
            'SecSSIMIndex': Sec_SSIM_index.avg,
            'ConPSNRIndex': Con_PSNR_index.avg,
            'SecPSNRIndex': Sec_PSNR_index.avg,
        }
        log_metrics(writer=self.writer, data_dict=validate_data_dict, step=epoch+1, prefix='validate')

        """
        val_log = "Validation[{:d}] ConAtt_loss: {:.6f} ConId_loss: {:.6f} ConMse_loss: {:.6f} ConLpips_loss: {:.6f} SecFeat_loss: {:.6f} SecMse_loss: {:.6f} SecLpips_loss: {:.6f} Sumlosses={:.6f} BatchTime: {:.4f}".format(
            epoch+1, Con_Att_loss.avg, Con_Id_loss.avg, Con_Mse_loss.avg, Con_Lpips_loss.avg, Sec_Feat_loss.avg, Sec_Mse_loss.avg, Sec_Lpips_loss.avg, Val_losses.avg, batch_time.avg
        )
        """
        """
        val_log = "Validation[{:d}] ConAtt_loss: {:.6f} ConId_loss: {:.6f} ConMse_loss: {:.6f} ConLpips_loss: {:.6f} ConSSIM_loss: {:.6f} ConPSNR_index: {:.6f} SecFeat_loss: {:.6f} SecMse_loss: {:.6f} SecLpips_loss: {:.6f} SecSSIM_loss: {:.6f} SecPSNR_index: {:.6f} Sumlosses={:.6f} BatchTime: {:.4f}".format(
            epoch+1, Con_Att_loss.avg, Con_Id_loss.avg, Con_Mse_loss.avg, Con_Lpips_loss.avg, Con_SSIM_loss.avg, Con_PSNR_index.avg, Sec_Feat_loss.avg, Sec_Mse_loss.avg, Sec_Lpips_loss.avg, Sec_SSIM_loss.avg, Sec_PSNR_index.avg, Val_losses.avg, batch_time.sum
        )
        """
        val_log = "Validation[{:d}] ConAtt_loss: {:.6f} ConId_loss: {:.6f} ConMse_loss: {:.6f} ConLpips_loss: {:.6f} ConSSIM_loss: {:.6f} ConPSNR_loss: {:.6f} ConSSIM_index: {:.6f} ConPSNR_index: {:.6f} SecFeat_loss: {:.6f} SecMse_loss: {:.6f} SecLpips_loss: {:.6f} SecSSIM_loss: {:.6f} SecPSNR_loss: {:.6f} SecSSIM_index: {:.6f} SecPSNR_index: {:.6f} Sumlosses={:.6f} BatchTime: {:.4f}".format(
            epoch+1, Con_Att_loss.avg, Con_Id_loss.avg, Con_Mse_loss.avg, Con_Lpips_loss.avg, Con_SSIM_loss.avg, Con_PSNR_loss.avg, Con_SSIM_index.avg, Con_PSNR_index.avg, Sec_Feat_loss.avg, Sec_Mse_loss.avg, Sec_Lpips_loss.avg, Sec_SSIM_loss.avg, Sec_PSNR_loss.avg, Sec_SSIM_index.avg, Sec_PSNR_index.avg, Val_losses.avg, batch_time.sum
        )
        print_log(info=val_log, log_path=self.args.logpath, console=True)

        #visualize_sihn_results(vis_dict=data_dict, dis_num=self.args.display_num, epoch=epoch+1, prefix='validation', save_dir=self.args.valpics_dir)
        visualize_results(vis_dict=data_dict, dis_num=self.args.display_num, epoch=epoch+1, prefix='validation', save_dir=self.args.valpics_dir)

        return Val_losses.avg, data_dict


    def running(self):
        print_log("Training is beginning .......................................................", self.args.logpath)

        self.global_train_steps = 0

        self.idencoder.eval()

        for epoch in range(self.args.max_epoch):
            """
            with torch.autograd.detect_anomaly(check_nan=True):
                self.training(epoch, self.train_cover_loader, self.secret_loader)
            """
            self.training(epoch, self.train_cover_loader, self.secret_loader)

            """
            self.att_scheduler.step()
            self.gen_scheduler.step()
            self.encoder_scheduler.step()                
            self.decoder_scheduler.step()
            self.fuser_scheduler.step()
            self.separator_scheduler.step()
            """
            
            if (epoch+1) % self.args.validation_interval == 0:
                with torch.no_grad():
                    validation_loss, data_dict = self.validation(epoch, self.val_cover_loader, self.secret_loader)
                
                stat_dict = {
                    'epoch': epoch + 1,
                    'id_state_dict': self.idencoder.state_dict(),
                    'att_state_dict': self.attencoder.state_dict(),
                    'gen_state_dict': self.generator.state_dict(),
                    'encoder_state_dict': self.encoder.state_dict(),
                    'decoder_state_dict': self.decoder.state_dict(),
                    'fuser_state_dict': self.fuser.state_dict(),
                    'separator_state_dict': self.separator.state_dict(),
                }
                self.save_checkpoint(stat_dict, is_best=False)

                if (self.best_loss is None) or (validation_loss < self.best_loss):
                    self.best_loss = validation_loss
                    self.save_checkpoint(stat_dict, is_best=True)

                    #visualize_sihn_results(vis_dict=data_dict, dis_num=self.args.display_num, epoch=epoch, prefix='best', save_dir=self.args.bestresults_dir)
                    visualize_results(vis_dict=data_dict, dis_num=self.args.display_num, epoch=epoch, prefix='best', save_dir=self.args.bestresults_dir)

        self.writer.close()
        print_log("Training finish .......................................................", self.args.logpath)


    def save_checkpoint(self, state, is_best):
        if is_best:
            torch.save(state['id_state_dict'], os.path.join(self.args.bestresults_dir, 'checkpoints', 'Id_best.pth'))
            torch.save(state['att_state_dict'], os.path.join(self.args.bestresults_dir, 'checkpoints', 'Att_best.pth'))
            torch.save(state['gen_state_dict'], os.path.join(self.args.bestresults_dir, 'checkpoints', 'Gen_best.pth'))
            torch.save(state['encoder_state_dict'], os.path.join(self.args.bestresults_dir, 'checkpoints', 'Encoder_best.pth'))
            torch.save(state['decoder_state_dict'], os.path.join(self.args.bestresults_dir, 'checkpoints', 'Decoder_best.pth'))
            torch.save(state['fuser_state_dict'], os.path.join(self.args.bestresults_dir, 'checkpoints', 'Fuser_best.pth'))
            torch.save(state['separator_state_dict'], os.path.join(self.args.bestresults_dir, 'checkpoints', 'Separator_best.pth'))
        else:
            torch.save(state, os.path.join(self.args.checkpoint_savedir, 'checkpoint.pth.tar'))



def main():
    args = TrainSihnAlphaOptions().parse()

    cur_time = time.strftime('%Y%m%d_H%H%M%S', time.localtime())
    args.output_dir = os.path.join(args.exp_dir, '{}'.format(cur_time))
    os.makedirs(args.output_dir, exist_ok=True)

    ### Initialize result directories and folders ###
    args.trainpics_dir = os.path.join(args.output_dir, 'TrainPics')
    os.makedirs(args.trainpics_dir, exist_ok=True)
    
    args.valpics_dir = os.path.join(args.output_dir, 'ValidatePics')
    os.makedirs(args.valpics_dir, exist_ok=True)
    
    args.checkpoint_savedir = os.path.join(args.output_dir, 'CheckPoints')
    os.makedirs(args.checkpoint_savedir, exist_ok=True)
    
    args.bestresults_dir = os.path.join(args.output_dir, 'BestResults')
    os.makedirs(args.bestresults_dir, exist_ok=True)
    os.makedirs(os.path.join(args.bestresults_dir, 'checkpoints'), exist_ok=True)

    args.tensorboardlogs_dir = os.path.join(args.output_dir, "TensorBoardLogs")
    os.makedirs(args.tensorboardlogs_dir, exist_ok=True)
    
    args.log_dir = os.path.join(args.output_dir, 'TrainingLogs')
    os.makedirs(args.log_dir, exist_ok=True)
    args.logpath = os.path.join(args.log_dir, 'train_log.txt')
    print_log(str(args), args.logpath, console=False)   

    args.args_dir = os.path.join(args.output_dir, 'Options')
    os.makedirs(args.args_dir, exist_ok=True) 

    print_log("[*]Exporting training results at {}".format(args.output_dir), args.logpath)
    
    args_dict = vars(args)
    with open(os.path.join(args.args_dir, 'opt.json'), 'w') as f:
        json.dump(args_dict, f, indent=4, sort_keys=True)

    train = Train(args)
    train.running()



if __name__ == '__main__':
	main()