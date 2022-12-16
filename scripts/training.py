import os
import sys
import time
import numpy as np

sys.path.append(".")
sys.path.append("..")

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from options.options import TrainOptions

from network.AAD import AADGenerator
from network.MAE import MLAttrEncoder
from network.Fuser import Fuser
from network.Separator import Separator
from network.Encoder import Encoder
from network.Decoder import Decoder

from criteria import loss_functions
from face_modules.model import Backbone

from utils.dataset import ImageDataset
from utils.common import visualize_results, print_log, alignment, l2_norm, log_metrics, AverageMeter




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
        # AAD Generator
        self.aadblocks = AADGenerator(c_id=512).to(self.args.device)
        try:
            self.aadblocks.load_state_dict(torch.load(os.path.join(self.args.aadblocks_dir, 'AAD_best.pth'), map_location=self.args.device), strict=True)
        except:
            print_log("[*]Training AAD Generator from scratch", self.args.logpath)
        
        # Att Encoder
        self.attencoder = MLAttrEncoder().to(self.args.device)
        try:
            self.attencoder.load_state_dict(torch.load(os.path.join(self.args.attencoder_dir, 'ATT_best.pth'), map_location=self.args.device), strict=True)
        except:
            print_log("[*]Training Attributes Encoder from scratch", self.args.logpath)

        # Id Encoder
        print_log("[*]Loading Face Recognition Model {} from {}".format(self.args.facenet_mode, self.args.facenet_dir), self.args.logpath)
        if self.args.facenet_mode == 'arcface':
            self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se').to(self.args.device)
            self.facenet.load_state_dict(torch.load(os.path.join(self.args.facenet_dir, 'model_ir_se50.pth'), map_location=self.args.device), strict=True)
        elif self.args.facenet_mode == 'circularface':
            self.facenet = Backbone(input_size=112, num_layers=100, drop_ratio=0.4, mode='ir', affine=False).to(self.args.device)
            self.facenet.load_state_dict(torch.load(os.path.join(self.args.facenet_dir, 'CurricularFace_Backbone.pth'), map_location=self.args.device), strict=True)
        else:
            raise ValueError("Invalid Face Recognition Model. Must be one of [arcface, CurricularFace]")
        
        # Fuser
        self.fuser = Fuser(latent_dim=self.args.latent_dim).to(self.args.device)
        try:
            self.fuser.load_state_dict(torch.load(os.path.join(self.args.fuser_dir, 'Fuser_best.pth'), map_location=self.args.device), strict=True)
        except:
            print_log("[*]Training Fuser from scratch", self.args.logpath)

        # Separator
        self.separator = Separator(latent_dim=self.args.latent_dim).to(self.args.device)
        try:
            self.separator.load_state_dict(torch.load(os.path.join(self.args.separator_dir, 'Separator_best.pth'), map_location=self.args.device), strict=True)
        except:
            print_log("[*]Training Separator from scratch", self.args.logpath)
        
        # Encoder
        self.encoder = Encoder(in_channels=3, latent_dim=self.args.latent_dim).to(self.args.device)
        try:
            self.encoder.load_state_dict(torch.load(os.path.join(self.args.encoder_dir, 'Encoder_best.pth'), map_location=self.args.device), strict=True)
        except:
            print_log("[*]Training Encoder from scratch", self.args.logpath)

        # Decoder
        self.decoder = Decoder(latent_dim=self.args.latent_dim).to(self.args.device)
        try:
            self.decoder.load_state_dict(torch.load(os.path.join(self.args.decoder_dir, 'Decoder_best.pth'), map_location=self.args.device), strict=True)
        except:
            print_log("[*]Training Decoder from scratch", self.args.logpath)

        ##### Initialize optimizers #####
        self.opt_aad = optim.Adam(self.aadblocks.parameters(), lr=self.args.lr_aad, betas=(0, 0.5))
        self.opt_att = optim.Adam(self.attencoder.parameters(), lr=self.args.lr_att, betas=(0, 0.5))
        self.opt_fuser = optim.Adam(self.fuser.parameters(), lr=self.args.lr_fuser, betas=(0, 0.5))
        self.opt_separator = optim.Adam(self.separator.parameters(), lr=self.args.lr_separator, betas=(0, 0.5))
        self.opt_encoder = optim.Adam(self.encoder.parameters(), lr=self.args.lr_encoder, betas=(0, 0.5))
        self.opt_decoder = optim.Adam(self.decoder.parameters(), lr=self.args.lr_decoder, betas=(0, 0.5))

        ##### Initialize loss functions #####
        self.att_loss = loss_functions.AttLoss().to(self.args.device)
        self.id_loss = loss_functions.IdLoss(self.args.idloss_mode).to(self.args.device)
        self.rec_con_loss = loss_functions.RecConLoss(self.args.recconloss_mode, self.args.device)
        self.rec_sec_loss = loss_functions.RecSecLoss(self.args.recsecloss_mode, self.args.device)
        self.feat_loss = loss_functions.FeatLoss(self.args.featloss_mode, self.args.device)

        ##### Initialize data loaders ##### 
        train_cover_transforms = transforms.Compose([
            transforms.Resize([self.args.image_size, self.args.image_size]),
            transforms.ToTensor()
        ])
        val_cover_transforms = transforms.Compose([
            transforms.Resize([self.args.image_size, self.args.image_size]),
            transforms.ToTensor()
        ])
        secret_transforms = transforms.Compose([
            transforms.Resize([self.args.image_size, self.args.image_size]),
            transforms.ToTensor()
        ])

        train_cover_dataset = ImageDataset(root=self.args.train_cover_dir, transforms=train_cover_transforms)
        val_cover_dataset = ImageDataset(root=self.args.val_cover_dir, transforms=val_cover_transforms)
        secret_dataset = ImageDataset(root=self.args.secret_dir, transforms=secret_transforms)
        
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
        secret = secret.to(self.args.device) # 1*256*256*3

        cover_att = self.attencoder(Xt=cover)

        cover_id = self.facenet(alignment(cover))
        cover_id_norm = l2_norm(cover_id)

        cover_id_fuse = cover_id_norm[:cover_id_norm.shape[0]//2]
        cover_id_ori = cover_id_norm[cover_id_fuse.shape[0]:]

        secret_feature = self.encoder(secret) # 1*256*256*3 -> 1*512
        secret_feature_norm = l2_norm(secret_feature)

        secret_feature_norm = secret_feature_norm.repeat(cover_id_fuse.shape[0], 1)

        secret_feature_null = torch.zeros(cover_id_norm.shape[0]-cover_id_fuse.shape[0], secret_feature_norm.shape[1]).to(self.args.device)
        secret_feature_ori = torch.cat((secret_feature_norm, secret_feature_null), dim=0)

        covsec_feature = self.fuser(cover_id_fuse, secret_feature_norm)
        covsec_feature = l2_norm(covsec_feature)

        fused_feature = torch.cat((covsec_feature, cover_id_ori), dim=0)

        container = self.aadblocks(inputs=(cover_att, fused_feature))

        container_id = self.facenet(alignment(container))
        container_id_norm = l2_norm(container_id)

        container_att = self.attencoder(Xt=container)

        container_id_fuse = container_id_norm[:cover_id_norm.shape[0]//2]

        secret_feature_ext = self.separator(container_id_fuse)
        secret_feature_ext = l2_norm(secret_feature_ext)

        secret_feature_null = torch.zeros(cover_id_norm.shape[0]-cover_id_fuse.shape[0], secret_feature_ext.shape[1]).to(self.args.device)
        secret_feature_rec = torch.cat((secret_feature_ext, secret_feature_null), dim=0)

        secret_rec = self.decoder(secret_feature_rec)

        secret_input = secret.repeat(cover_id_fuse.shape[0], 1, 1, 1)
        secret_null = torch.zeros(cover_id_norm.shape[0]-cover_id_fuse.shape[0], secret.shape[1], secret.shape[2], secret.shape[3]).to(self.args.device)
        secret_batch = torch.cat((secret_input, secret_null), dim=0)

        ##### Collect results ##### 
        data_dict = {
            'cover': cover,
            'container': container,
            'secret': secret_batch,
            'secret_rec': secret_rec,
            'cover_id': cover_id_norm,
            'fused_feature': fused_feature,
            'container_id': container_id_norm,
            'secret_feature_ori': secret_feature_ori,
            'secret_feature_rec': secret_feature_rec,
            'cover_att': cover_att,
            'container_att': container_att,
        }

        return data_dict


    def training(self, epoch, cover_loader, secret_loader):
        batch_time = AverageMeter()
        Att_loss = AverageMeter()
        Id_loss = AverageMeter()
        Rec_con_loss = AverageMeter()
        Rec_sec_loss = AverageMeter()
        Feat_loss = AverageMeter()
        
        Train_losses = AverageMeter()

        start_time = time.time()

        secret_iterator = iter(secret_loader)
        for train_iter, cover_batch in enumerate(cover_loader):
            try:
                secret_batch = next(secret_iterator)
            except StopIteration:
                secret_iterator = iter(secret_loader)
                secret_batch = next(secret_iterator)

            #cover_batch = cover_batch.to(self.args.device)
            #secret_batch = secret_batch.to(self.args.device)

            ##### Training #####
            self.facenet.eval()

            self.aadblocks.train()
            self.attencoder.train()
            self.fuser.train()
            self.separator.train()
            self.encoder.train()
            self.decoder.train()

            data_dict = self.forward_pass(cover_batch, secret_batch)

            loss_att = self.att_loss(data_dict['container_att'], data_dict['cover_att'])
            # 可以给 Id 设置一个比例，一部分约束 fused feature 和 container id，另外一部分约束 cover id 和 container id
            loss_id = self.id_loss(data_dict['fused_feature'], data_dict['container_id'])
            loss_con_rec = self.rec_con_loss(data_dict['container'], data_dict['cover'])
            loss_sec_rec = self.rec_sec_loss(data_dict['secret_rec'], data_dict['secret'])
            loss_feat = self.feat_loss(data_dict['secret_feature_rec'], data_dict['secret_feature_ori'])

            Sum_train_losses = self.args.att_lambda*loss_att + self.args.id_lambda*loss_id + self.args.rec_con_lambda*loss_con_rec + self.args.rec_sec_lambda*loss_sec_rec + self.args.feat_lambda*loss_feat

            self.opt_aad.zero_grad()
            self.opt_att.zero_grad()
            self.opt_fuser.zero_grad()
            self.opt_separator.zero_grad()
            self.opt_encoder.zero_grad()
            self.opt_decoder.zero_grad()

            Sum_train_losses.backward()

            self.opt_aad.step()
            self.opt_att.step()
            self.opt_fuser.step()
            self.opt_separator.step()
            self.opt_encoder.step()
            self.opt_decoder.step()

            ##### Log losses and computation time #####
            Att_loss.update(loss_att.item(), self.args.train_bs)
            Id_loss.update(loss_id.item(), self.args.train_bs)
            Rec_con_loss.update(loss_con_rec.item(), self.args.train_bs)
            Rec_sec_loss.update(loss_sec_rec.item(), self.args.train_bs)
            Feat_loss.update(loss_feat.item(), self.args.train_bs)

            Train_losses.update(Sum_train_losses.item(), self.args.train_bs)

            batch_time.update(time.time()-start_time)
            start_time = time.time()

            train_data_dict = {
                'AttLoss': Att_loss.avg,
                'IdLoss': Id_loss.avg,
                'RecConLoss': Rec_con_loss.avg,
                'RecSecLoss': Rec_sec_loss.avg,
                'FeatLoss': Feat_loss.avg,
                'SumTrainLosses': Train_losses.avg
            }

            ##### Board and log losses, and visualize results #####
            if (self.global_train_steps+1) % self.args.board_interval == 0:
                train_log = "[{:d}/{:d}][Iteration: {:05d}][Steps: {:05d}] Att_loss: {:.6f} Id_loss: {:.6f} Rec_con_loss: {:.6f} Rec_sec_loss: {:.6f} Feat_loss: {:.6f} Sumlosses={:.6f} BatchTime: {:.4f}".format(
                    epoch+1, self.args.max_epoch, train_iter+1, self.global_train_steps+1, Att_loss.val, Id_loss.val, Rec_con_loss.val, Rec_sec_loss.val, Feat_loss.val, Train_losses.val, batch_time.val
                )
                print_log(info=train_log, log_path=self.args.logpath, console=True)
                log_metrics(writer=self.writer, data_dict=train_data_dict, step=self.global_train_steps+1, prefix='train')

            if self.global_train_steps == 0 or (self.global_train_steps+1) % self.args.image_interval == 0:
                visualize_results(vis_dict=data_dict, dis_num=self.args.display_num, epoch=epoch+1, prefix='train', save_dir=self.args.trainpics_dir, iter=train_iter+1, step=self.global_train_steps+1)
            
            self.global_train_steps += 1
            
            if train_iter == self.args.max_train_iters-1:
                break


    def validation(self, epoch, cover_loader, secret_loader):
        self.facenet.eval()
        self.aadblocks.eval()
        self.attencoder.eval()
        self.fuser.eval()
        self.separator.eval()
        self.encoder.eval()
        self.decoder.eval()

        batch_time = AverageMeter()

        Att_loss = AverageMeter()
        Id_loss = AverageMeter()
        Rec_con_loss = AverageMeter()
        Rec_sec_loss = AverageMeter()
        Feat_loss = AverageMeter()
        
        Val_losses = AverageMeter()        

        start_time = time.time()

        secret_iterator = iter(secret_loader)
        for val_iter, cover_batch in enumerate(cover_loader):
            try:
                secret_batch = next(secret_iterator)
            except StopIteration:
                secret_iterator = iter(secret_loader)
                secret_batch = next(secret_iterator)
            
            cover_batch = cover_batch.to(self.args.device)
            secret_batch = secret_batch.to(self.args.device)

            data_dict = self.forward_pass(cover_batch, secret_batch)

            # Calculate losses
            loss_att = self.att_loss(data_dict['container_att'], data_dict['cover_att'])
            loss_id = self.id_loss(data_dict['fused_feature'], data_dict['container_id'])
            loss_con_rec = self.rec_con_loss(data_dict['container'], data_dict['cover'])
            loss_sec_rec = self.rec_sec_loss(data_dict['secret_rec'], data_dict['secret'])
            loss_feat = self.feat_loss(data_dict['secret_feature_rec'], data_dict['secret_feature'])

            sum_val_loss = self.args.att_lambda*loss_att + self.args.id_lambda*loss_id + self.args.rec_con_lambda*loss_con_rec + self.args.rec_sec_lambda*loss_sec_rec + self.args.feat_lambda*loss_feat

            Att_loss.update(loss_att.item(), self.args.train_bs)
            Id_loss.update(loss_id.item(), self.args.train_bs)
            Rec_con_loss.update(loss_con_rec.item(), self.args.train_bs)
            Rec_sec_loss.update(loss_sec_rec.item(), self.args.train_bs)
            Feat_loss.update(loss_feat.item(), self.args.train_bs)
            
            Val_losses.update(sum_val_loss.item(), self.args.train_bs)

            batch_time.update(time.time() - start_time)
            start_time = time.time()

            if val_iter == self.args.max_val_iters-1:
                break

        validate_data_dict = {
            'AttLoss': Att_loss.avg,
            'IdLoss': Id_loss.avg,
            'RecConLoss': Rec_con_loss.avg,
            'RecSecLoss': Rec_sec_loss.avg,
            'FeatLoss': Feat_loss.avg,
            'SumValidateLosses': Val_losses.avg
        }
        log_metrics(writer=self.writer, data_dict=validate_data_dict, step=epoch+1, prefix='validate')

        val_log = "Validation[{:d}] Att_loss: {:.6f} Id_loss: {:.6f} Rec_con_loss: {:.6f} Rec_sec_loss: {:.6f} Feat_loss: {:.6f} Sumlosses={:.6f} BatchTime: {:.4f}".format(
            epoch+1, Att_loss.avg, Id_loss.avg, Rec_con_loss.avg, Rec_sec_loss.avg, Feat_loss.avg, Val_losses.avg, batch_time.avg
        )
        print_log(info=val_log, log_path=self.args.logpath, console=True)

        visualize_results(vis_dict=data_dict, dis_num=self.args.display_num, epoch=epoch+1, prefix='validation', save_dir=self.args.valpics_dir)

        return Val_losses.avg, data_dict


    def running(self):
        print_log("Training is beginning .......................................................", self.args.logpath)

        self.global_train_steps = 0

        for epoch in range(self.args.max_epoch):
            self.training(epoch, self.train_cover_loader, self.secret_loader)
            
            if epoch % self.args.validation_interval == 0:
                with torch.no_grad():
                    validation_loss, data_dict = self.validation(epoch, self.val_cover_loader, self.secret_loader)
                
                stat_dict = {
                    'epoch': epoch + 1,
                    'aad_state_dict': self.aadblocks.state_dict(),
                    'att_state_dict': self.attencoder.state_dict(),
                    'fuser_state_dict': self.fuser.state_dict(),
                    'separator_state_dict': self.separator.state_dict(),
                    'encoder_state_dict': self.encoder.state_dict(),
                    'decoder_state_dict': self.decoder.state_dict(),
                }
                self.save_checkpoint(stat_dict, is_best=False)

                if (self.best_loss is None) or (validation_loss < self.best_loss):
                    self.best_loss = validation_loss
                    self.save_checkpoint(stat_dict, is_best=True)
                    
                    visualize_results(vis_dict=data_dict, dis_num=self.args.display_num, epoch=epoch, prefix='best', save_dir=self.args.bestresults_dir)
        
        self.writer.close()
        print_log("Training finish .......................................................", self.args.logpath)


    def save_checkpoint(self, state, is_best):
        if is_best:
            torch.save(state['aad_state_dict'], os.path.join(self.args.bestresults_dir, 'checkpoints', 'AAD_best.pth'))
            torch.save(state['att_state_dict'], os.path.join(self.args.bestresults_dir, 'checkpoints', 'ATT_best.pth'))
            torch.save(state['fuser_state_dict'], os.path.join(self.args.bestresults_dir, 'checkpoints', 'Fuser_best.pth'))
            torch.save(state['separator_state_dict'], os.path.join(self.args.bestresults_dir, 'checkpoints', 'Separator_best.pth'))
            torch.save(state['encoder_state_dict'], os.path.join(self.args.bestresults_dir, 'checkpoints', 'Encoder_best.pth'))
            torch.save(state['decoder_state_dict'], os.path.join(self.args.bestresults_dir, 'checkpoints', 'Decoder_best.pth'))
        else:
            torch.save(state, os.path.join(self.args.checkpoints_dir, 'checkpoint.pth.tar'))



def main():
    args = TrainOptions().parse()

    cur_time = time.strftime('%Y%m%d_H%H%M%S', time.localtime())
    args.output_dir = os.path.join(args.exp_dir, '{}'.format(cur_time))
    os.makedirs(args.output_dir, exist_ok=True)

    ### Initialize result directories and folders ###
    args.trainpics_dir = os.path.join(args.output_dir, 'TrainPics')
    os.makedirs(args.trainpics_dir, exist_ok=True)
    
    args.valpics_dir = os.path.join(args.output_dir, 'ValidatePics')
    os.makedirs(args.valpics_dir, exist_ok=True)
    
    args.checkpoints_dir = os.path.join(args.output_dir, 'CheckPoints')
    os.makedirs(args.checkpoints_dir, exist_ok=True)
    
    args.bestresults_dir = os.path.join(args.output_dir, 'BestResults')
    os.makedirs(args.bestresults_dir, exist_ok=True)
    os.makedirs(os.path.join(args.bestresults_dir, 'checkpoints'), exist_ok=True)

    args.tensorboardlogs_dir = os.path.join(args.output_dir, "TensorBoardLogs")
    os.makedirs(args.tensorboardlogs_dir, exist_ok=True)
    
    args.log_dir = os.path.join(args.output_dir, 'TrainingLogs')
    os.makedirs(args.log_dir, exist_ok=True)
    args.logpath = os.path.join(args.log_dir, 'train_log.txt')
    print_log(str(args), args.logpath, console=False)    

    print_log("[*]Exporting training results at {}".format(args.output_dir), args.logpath)

    train = Train(args)
    train.running()



if __name__ == '__main__':
	main()