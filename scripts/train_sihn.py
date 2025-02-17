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

from options.options import TrainSihnOptions

from network.DisentanglementEncoder import DisentanglementEncoder
from network.AAD import AADGenerator
from network.Fuser import Fuser
from network.Separator import Separator
from network.Encoder import PspEncoder, MappingNetwork
from stylegan2.model import Generator

from criteria import loss_functions

from utils.dataset import ImageDataset
from utils.common import weights_init, visualize_results, print_log, log_metrics, l2_norm, AverageMeter




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
        # Disentanglement Encoder
        self.disentangler = DisentanglementEncoder(latent_dim=self.args.latent_dim).to(self.args.device)
        try:
            self.disentangler.load_state_dict(torch.load(os.path.join(self.args.checkpoint_dir, 'Dis_best.pth'), map_location=self.args.device), strict=True)
            print_log("[*]Successfully loaded Disentangler's pre-trained model", self.args.logpath)
        except:
            raise ValueError("[*]Unable to load Disentangler's pre-trained model")

        # AAD Generator
        self.generator= AADGenerator(c_id=512).to(self.args.device)
        try:
            self.generator.load_state_dict(torch.load(os.path.join(self.args.checkpoint_dir, 'Gen_best.pth'), map_location=self.args.device), strict=True)
            print_log("[*]Successfully loaded Generator's pre-trained model", self.args.logpath)
        except:
            raise ValueError("[*]Unable to load Generator's pre-trained model")
        
        # Encoder 
        self.encoder = PspEncoder(num_layers=50, mode='ir_se').to(self.args.device)
        try:
            self.encoder.load_state_dict(torch.load(os.path.join(self.args.checkpoint_dir, 'Encoder_best.pth'), map_location=self.args.device), strict=True)
            print_log("[*]Successfully loaded Encoder's pre-trained model", self.args.logpath)
        except:
            raise ValueError("[*]Unable to load Encoder's pre-trained model")

        # Mapper
        self.mapper = MappingNetwork().to(self.args.device)
        try:
            self.mapper.load_state_dict(torch.load(os.path.join(self.args.checkpoint_dir, 'Mapper_best.pth'), map_location=self.args.device), strict=True)
            print_log("[*]Successfully loaded Mapper's pre-trained model", self.args.logpath)
        except:
            print_log("[*]Training Mapper from scratch", self.args.logpath)
            self.mapper.apply(weights_init)

        # Decoder 
        #default size=1024, style_dim=512, n_mlp=8
        self.decoder = Generator(size=self.args.image_size, style_dim=self.args.latent_dim, n_mlp=8).to(self.args.device)
        try:
            self.decoder.load_state_dict(torch.load(os.path.join(self.args.checkpoint_dir, 'Decoder_best.pth'), map_location=self.args.device), strict=True)
            print_log("[*]Successfully loaded Decoder's pre-trained model", self.args.logpath)
        except:
            raise ValueError("[*]Unable to load Decoder's pre-trained model")

        # Fuser 
        self.fuser = Fuser(latent_dim=self.args.latent_dim).to(self.args.device)
        try:
            self.fuser.load_state_dict(torch.load(os.path.join(self.args.checkpoint_dir, 'Fuser_best.pth')))
            print_log("[*]Successfully loaded Fuser's pre-trained model", self.args.logpath)
        except:
            print_log("[*]Training Fuser from scratch", self.args.logpath)
            self.fuser.apply(weights_init)

        # Separator 
        self.separator = Separator(latent_dim=self.args.latent_dim).to(self.args.device)
        try:
            self.separator.load_state_dict(torch.load(os.path.join(self.args.checkpoint_dir, 'Separator_best.pth')))
            print_log("[*]Successfully loaded Separator's pre-trained model", self.args.logpath)
        except:
            print_log("[*]Training Separator from scratch", self.args.logpath)
            self.separator.apply(weights_init)


        ##### Initialize optimizers #####
        #self.dis_optim = optim.Adam(self.disentangler.parameters(), lr=self.args.dis_lr, betas=(0.5, 0.999))
        #self.gen_optim = optim.Adam(self.generator.parameters(), lr=self.args.gen_lr, betas=(0.5, 0.999))
        #self.encoder_optim = optim.Adam(self.encoder.parameters(), lr=self.args.encoder_lr, betas=(0.5, 0.999))
        #self.decoder_optim = optim.Adam(self.decoder.parameters(), lr=self.args.decoder_lr, betas=(0.5, 0.999))
        self.fuser_optim = optim.Adam(self.fuser.parameters(), lr=self.args.fuser_lr, betas=(0.5, 0.999))
        self.separator_optim = optim.Adam(self.separator.parameters(), lr=self.args.separator_lr, betas=(0.5, 0.999))
        
        #self.dis_scheduler = optim.lr_scheduler.ExponentialLR(optimizer=self.dis_optim, gamma=0.9, last_epoch=-1, verbose=True)
        #self.gen_scheduler = optim.lr_scheduler.ExponentialLR(optimizer=self.gen_optim, gamma=0.9, last_epoch=-1, verbose=True)
        #self.encoder_scheduler = optim.lr_scheduler.ExponentialLR(optimizer=self.encoder_optim, gamma=0.9, last_epoch=-1, verbose=True)
        #self.decoder_scheduler = optim.lr_scheduler.ExponentialLR(optimizer=self.decoder_optim, gamma=0.9, last_epoch=-1, verbose=True)
        #self.fuser_scheduler = optim.lr_scheduler.ExponentialLR(optimizer=self.fuser_optim, gamma=0.9, last_epoch=-1, verbose=True)
        #self.separator_scheduler = optim.lr_scheduler.ExponentialLR(optimizer=self.separator_optim, gamma=0.9, last_epoch=-1, verbose=True)
        
        #self.fuser_scheduler = optim.lr_scheduler.StepLR(optimizer=self.fuser_optim, step_size=self.args.step_size, gamma=0.2, last_epoch=-1,verbose=True)
        #self.separator_scheduler = optim.lr_scheduler.StepLR(optimizer=self.fuser_optim, step_size=self.args.step_size, gamma=0.2, last_epoch=-1, verbose=True)


        ##### Initialize loss functions #####
        self.con_att_loss = loss_functions.AttLoss().to(self.args.device).eval()
        self.con_id_loss = loss_functions.IdLoss(self.args.conidloss_mode).to(self.args.device).eval()
        self.con_rec_loss = loss_functions.RecConLoss(self.args.conrecloss_mode, self.args.device).eval()
        self.sec_feat_loss = loss_functions.FeatLoss(self.args.secfeatloss_mode, self.args.device).eval()
        self.sec_mse_loss = loss_functions.RecSecLoss('l2', self.args.device).eval()
        self.sec_lpips_loss = loss_functions.RecSecLoss('lpips', self.args.device).eval()


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
        cover_id, cover_att = self.disentangler(cover)

        secret = secret.to(self.args.device)
        secret_ori = secret.repeat(cover.shape[0]//2, 1, 1, 1)
        secret_null = torch.zeros(cover.shape[0] - secret_ori.shape[0], secret_ori.shape[1], secret_ori.shape[2], secret_ori.shape[3]).to(self.args.device)
        secret_input = torch.cat((secret_ori, secret_null), dim=0)

        secret_feature_input = self.encoder(secret_input)

        input_feature = self.fuser(cover_id, secret_feature_input)

        container = self.generator(inputs=(cover_att, input_feature))

        container_id, container_att = self.disentangler(container)

        secret_feature_output = self.separator(container_id)

        secret_feature_output_plus = self.mapper(secret_feature_output)

        secret_rec, _ = self.decoder(
            styles=[secret_feature_output_plus],
            input_is_latent=True,
            randomize_noise=True,
            return_latents=False,
        )

        #secret_feature_rec = self.encoder(secret_rec)

        ##### Collect results ##### 
        data_dict = {
            'cover': cover,
            'container': container,
            'secret_ori': secret_input,
            'secret_rec': secret_rec,
            'cover_id': cover_id,
            'input_feature': input_feature,
            'container_id': container_id,
            #'secret_feature_rec': secret_feature_rec,
            'secret_feature_input': secret_feature_input,
            'secret_feature_output': secret_feature_output,
            'cover_att': cover_att,
            'container_att': container_att,
        }

        return data_dict


    def training(self, epoch, cover_loader, secret_loader):
        self.fuser.train()
        self.separator.train()

        batch_time = AverageMeter()

        Con_Att_loss = AverageMeter()
        Con_Id_loss = AverageMeter()
        Con_Rec_loss = AverageMeter()
        Sec_Feat_loss = AverageMeter()
        Sec_Mse_loss = AverageMeter()
        Sec_Lpips_loss = AverageMeter()

        Train_losses = AverageMeter()

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
            loss_con_rec = self.con_rec_loss(data_dict['container'], data_dict['cover'])
            #loss_sec_feat = 0.5*self.sec_feat_loss(data_dict['secret_feature_rec'], data_dict['secret_feature_input']) + 0.5*self.sec_feat_loss(data_dict['secret_feature_output'], data_dict['secret_feature_input'])
            loss_sec_feat = self.sec_feat_loss(data_dict['secret_feature_output'], data_dict['secret_feature_input'])
            loss_sec_mse = self.sec_mse_loss(data_dict['secret_rec'], data_dict['secret_ori'])
            loss_sec_lpips = self.sec_lpips_loss(data_dict['secret_rec'], data_dict['secret_ori'])

            Sum_train_losses = self.args.con_att_lambda*loss_con_att + self.args.con_id_lambda*loss_con_id + self.args.con_rec_lambda*loss_con_rec \
            + self.args.sec_feat_lambda*loss_sec_feat + self.args.sec_mse_lambda*loss_sec_mse + self.args.sec_lpips_lambda*loss_sec_lpips

            #self.dis_optim.zero_grad()
            #self.gen_optim.zero_grad()
            #self.encoder_optim.zero_grad()
            #self.decoder_optim.zero_grad()
            self.fuser_optim.zero_grad()
            self.separator_optim.zero_grad()

            Sum_train_losses.backward()

            #self.dis_optim.step()
            #self.gen_optim.step()
            #self.encoder_optim.step()
            #self.decoder_optim.step()
            self.fuser_optim.step()
            self.separator_optim.step()
            
            ##### Log losses and computation time #####
            Con_Att_loss.update(loss_con_att.item(), self.args.train_bs)
            Con_Id_loss.update(loss_con_id.item(), self.args.train_bs)
            Con_Rec_loss.update(loss_con_rec.item(), self.args.train_bs)
            Sec_Feat_loss.update(loss_sec_feat.item(), self.args.train_bs)
            Sec_Mse_loss.update(loss_sec_mse.item(), self.args.train_bs)
            Sec_Lpips_loss.update(loss_sec_lpips.item(), self.args.train_bs)
            Train_losses.update(Sum_train_losses.item(), self.args.train_bs)

            batch_time.update(time.time()-start_time)
            start_time = time.time()

            train_data_dict = {
                'ConAttLoss': Con_Att_loss.avg,
                'ConIdLoss': Con_Id_loss.avg,
                'ConRecLoss': Con_Rec_loss.avg,
                'SecFeatLoss': Sec_Feat_loss.avg,
                'SecMseLoss': Sec_Mse_loss.avg,
                'SecLpipsLoss': Sec_Lpips_loss.avg,
                'SumTrainLosses': Train_losses.avg,
            }

            ##### Board and log losses, and visualize results #####
            if (self.global_train_steps+1) % self.args.board_interval == 0:
                train_log = "[{:d}/{:d}][Iteration: {:05d}][Steps: {:05d}] ConAtt_loss: {:.6f} ConId_loss: {:.6f} ConRec_loss: {:.6f} SecFeat_loss: {:.6f} SecMse_loss: {:.6f} SecLpips_loss: {:.6f} Sumlosses={:.6f} BatchTime: {:.4f}".format(
                    epoch+1, self.args.max_epoch, train_iter+1, self.global_train_steps+1, Con_Att_loss.val, Con_Id_loss.val, Con_Rec_loss.val, Sec_Feat_loss.val, Sec_Mse_loss.val, Sec_Lpips_loss.val, Train_losses.val, batch_time.val
                )
                print_log(info=train_log, log_path=self.args.logpath, console=True)
                log_metrics(writer=self.writer, data_dict=train_data_dict, step=self.global_train_steps+1, prefix='train')

            if self.global_train_steps == 0 or (self.global_train_steps+1) % self.args.image_interval == 0:
                visualize_results(vis_dict=data_dict, dis_num=self.args.display_num, epoch=epoch+1, prefix='train', save_dir=self.args.trainpics_dir, iter=train_iter+1, step=self.global_train_steps+1)
            
            self.global_train_steps += 1
            
            if train_iter == self.args.max_train_iters-1:
                break
        
        train_epoch_log = "Training[{:d}/{:d}] ConAtt_loss: {:.6f} ConId_loss: {:.6f} ConRec_loss: {:.6f} SecFeat_loss: {:.6f} SecMse_loss: {:.6f} SecLpips_loss: {:.6f} Sumlosses={:.6f} BatchTime: {:.4f}".format(
            epoch+1, self.args.max_epoch, Con_Att_loss.avg, Con_Id_loss.avg, Con_Rec_loss.avg, Sec_Feat_loss.avg, Sec_Mse_loss.avg, Sec_Lpips_loss.avg, Train_losses.avg, batch_time.sum
        )
        print_log(info=train_epoch_log, log_path=self.args.logpath, console=True)


    def validation(self, epoch, cover_loader, secret_loader):
        self.fuser.eval()
        self.separator.eval()

        batch_time = AverageMeter()

        Con_Att_loss = AverageMeter()
        Con_Id_loss = AverageMeter()
        Con_Rec_loss = AverageMeter()
        Sec_Feat_loss = AverageMeter()
        Sec_Mse_loss = AverageMeter()
        Sec_Lpips_loss = AverageMeter()
        Val_losses = AverageMeter()

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
            loss_con_rec = self.con_rec_loss(data_dict['container'], data_dict['cover'])
            #loss_sec_feat = 0.5*self.sec_feat_loss(data_dict['secret_feature_rec'], data_dict['secret_feature_input']) + 0.5*self.sec_feat_loss(data_dict['secret_feature_output'], data_dict['secret_feature_input'])
            loss_sec_feat = self.sec_feat_loss(data_dict['secret_feature_output'], data_dict['secret_feature_input'])
            loss_sec_mse = self.sec_mse_loss(data_dict['secret_rec'], data_dict['secret_ori'])
            loss_sec_lpips = self.sec_lpips_loss(data_dict['secret_rec'], data_dict['secret_ori'])

            sum_val_loss = self.args.con_att_lambda*loss_con_att + self.args.con_id_lambda*loss_con_id + self.args.con_rec_lambda*loss_con_rec \
            + self.args.sec_feat_lambda*loss_sec_feat + self.args.sec_mse_lambda*loss_sec_mse + self.args.sec_lpips_lambda*loss_sec_lpips

            Con_Att_loss.update(loss_con_att.item(), self.args.train_bs)
            Con_Id_loss.update(loss_con_id.item(), self.args.train_bs)
            Con_Rec_loss.update(loss_con_rec.item(), self.args.train_bs)
            Sec_Feat_loss.update(loss_sec_feat.item(), self.args.train_bs)
            Sec_Mse_loss.update(loss_sec_mse.item(), self.args.train_bs)
            Sec_Lpips_loss.update(loss_sec_lpips.item(), self.args.train_bs)
            Val_losses.update(sum_val_loss.item(), self.args.train_bs)

            batch_time.update(time.time() - start_time)
            start_time = time.time()

            if val_iter == self.args.max_val_iters-1:
                break

        validate_data_dict = {
            'ConAttLoss': Con_Att_loss.avg,
            'ConIdLoss': Con_Id_loss.avg,
            'ConRecLoss': Con_Rec_loss.avg,
            'SecFeatLoss': Sec_Feat_loss.avg,
            'SecMseLoss': Sec_Mse_loss.avg,
            'SecLpipsLoss': Sec_Lpips_loss.avg,
            'SumValidateLosses': Val_losses.avg,
        }
        log_metrics(writer=self.writer, data_dict=validate_data_dict, step=epoch+1, prefix='validate')

        val_log = "Validation[{:d}] ConAtt_loss: {:.6f} ConId_loss: {:.6f} ConRec_loss: {:.6f} SecFeat_loss: {:.6f} SecMse_loss: {:.6f} SecLpips_loss: {:.6f} Sumlosses={:.6f} BatchTime: {:.4f}".format(
            epoch+1, Con_Att_loss.avg, Con_Id_loss.avg, Con_Rec_loss.avg, Sec_Feat_loss.avg, Sec_Mse_loss.avg, Sec_Lpips_loss.avg, Val_losses.avg, batch_time.avg
        )
        print_log(info=val_log, log_path=self.args.logpath, console=True)

        visualize_results(vis_dict=data_dict, dis_num=self.args.display_num, epoch=epoch+1, prefix='validation', save_dir=self.args.valpics_dir)

        return Val_losses.avg, data_dict


    def running(self):
        print_log("Training is beginning .......................................................", self.args.logpath)

        self.global_train_steps = 0

        self.disentangler.eval()
        self.generator.eval()
        self.encoder.eval()
        self.decoder.eval()
        self.mapper.eval()

        for epoch in range(self.args.max_epoch):
            self.training(epoch, self.train_cover_loader, self.secret_loader)
            
            if (epoch+1) % self.args.validation_interval == 0:
                with torch.no_grad():
                    validation_loss, data_dict = self.validation(epoch, self.val_cover_loader, self.secret_loader)
                
                #self.dis_scheduler.step()
                #self.gen_scheduler.step()
                #self.encoder_scheduler.step()
                #self.decoder_scheduler.step()
                #self.fuser_scheduler.step()
                #self.separator_scheduler.step()
                
                stat_dict = {
                    'epoch': epoch + 1,
                    #'dis_state_dict': self.disentangler.state_dict(),
                    #'gen_state_dict': self.generator.state_dict(),
                    #'encoder_state_dict': self.encoder.state_dict(),
                    #'decoder_state_dict': self.decoder.state_dict(),
                    'fuser_state_dict': self.fuser.state_dict(),
                    'separator_state_dict': self.separator.state_dict(),
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
            #torch.save(state['dis_state_dict'], os.path.join(self.args.bestresults_dir, 'checkpoints', 'Dis_best.pth'))
            #torch.save(state['gen_state_dict'], os.path.join(self.args.bestresults_dir, 'checkpoints', 'Gen_best.pth'))
            #torch.save(state['encoder_state_dict'], os.path.join(self.args.bestresults_dir, 'checkpoints', 'Encoder_best.pth'))
            #torch.save(state['decoder_state_dict'], os.path.join(self.args.bestresults_dir, 'checkpoints', 'Decoder_best.pth'))
            torch.save(state['fuser_state_dict'], os.path.join(self.args.bestresults_dir, 'checkpoints', 'Fuser_best.pth'))
            torch.save(state['separator_state_dict'], os.path.join(self.args.bestresults_dir, 'checkpoints', 'Separator_best.pth'))
        else:
            torch.save(state, os.path.join(self.args.checkpoint_savedir, 'checkpoint.pth.tar'))



def main():
    args = TrainSihnOptions().parse()

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

    print_log("[*]Exporting training results at {}".format(args.output_dir), args.logpath)

    train = Train(args)
    train.running()



if __name__ == '__main__':
	main()