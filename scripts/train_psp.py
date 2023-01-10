import os
import sys
import time
import numpy as np

sys.path.append(".")
sys.path.append("..")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from options.options import TrainPspOptions

from network.Encoder import BackboneEncoderUsingLastLayerIntoWPlus
from stylegan2.model import Generator

from criteria.lpips.lpips import LPIPS

from utils.dataset import ImageDataset
from utils.common import weights_init, visualize_psp_results, print_log, log_metrics, AverageMeter




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
        # Encoder
        self.encoder = BackboneEncoderUsingLastLayerIntoWPlus(50, 'ir_se').to(self.args.device)
        try:
            self.encoder.load_state_dict(torch.load(os.path.join(self.args.checkpoint_dir, 'Encoder_best.pth'), map_location=self.args.device), strict=True)
        except:
            print_log("[*]Training Encoder from scratch", self.args.logpath)
            self.encoder.apply(weights_init)

        # Decoder
        #self.decoder = Decoder(latent_dim=self.args.latent_dim).to(self.args.device)
        self.decoder = Generator(256, 512, 8).to(self.args.device)
        try:
            self.decoder.load_state_dict(torch.load(os.path.join(self.args.checkpoint_dir, 'Decoder_best.pth'), map_location=self.args.device), strict=True)
        except:
            print_log("[*]Training Decoder from scratch", self.args.logpath)
            self.decoder.apply(weights_init)

        ##### Initialize optimizers #####
        self.encoder_optim = optim.Adam(self.encoder.parameters(), lr=self.args.encoder_lr, betas=(0.5, 0.999), weight_decay=0)
        self.decoder_optim = optim.Adam(self.decoder.parameters(), lr=self.args.decoder_lr, betas=(0.5, 0.999), weight_decay=0)

        self.encoder_scheduler = optim.lr_scheduler.StepLR(optimizer=self.encoder_optim, step_size=10, gamma=0.2, last_epoch=-1, verbose=True)
        self.decoder_scheduler = optim.lr_scheduler.StepLR(optimizer=self.decoder_optim, step_size=10, gamma=0.2, last_epoch=-1, verbose=True)

        ##### Initialize loss functions #####
        self.mse_loss = nn.MSELoss().to(self.args.device).eval()
        self.lpips_loss = LPIPS(net_type='alex').to(self.args.device).eval()

        ##### Initialize data loaders ##### 
        train_transforms = transforms.Compose([
            transforms.Resize([self.args.image_size, self.args.image_size]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ])
        val_transforms = transforms.Compose([
            transforms.Resize([self.args.image_size, self.args.image_size]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ])

        train_dataset = ImageDataset(root=self.args.train_dir, transforms=train_transforms)
        val_dataset = ImageDataset(root=self.args.val_dir, transforms=val_transforms)
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.train_bs,
            shuffle=True,
            num_workers=int(self.args.num_workers),
            drop_last=True
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.args.train_bs,
            shuffle=True,
            num_workers=int(self.args.num_workers),
            drop_last=True
        )

        ##### Initialize logger #####
        self.writer = SummaryWriter(log_dir=self.args.tensorboardlogs_dir)
        self.best_loss = None


    def forward_pass(self, image_ori):
        image_ori = image_ori.to(self.args.device)
        
        image_feature = self.encoder(image_ori)
        #image_feature, image_mu, image_logsigma2 = self.encoder(image_ori)
        
        image_rec, _ = self.decoder(
            styles=[image_feature],
            input_is_latent=True,
            randomize_noise=True,
            return_latents=False,
            )

        ##### Collect results ##### 
        data_dict = {
            'image_ori': image_ori,
            'image_rec': image_rec,
        }

        return data_dict


    def training(self, epoch, image_loader):
        self.encoder.train()
        self.decoder.train()

        batch_time = AverageMeter()

        MSE_loss = AverageMeter()
        LPIPS_loss = AverageMeter()
        TrainLosses = AverageMeter()

        start_time = time.time()

        for train_iter, image_batch in enumerate(image_loader):
            ##### Training #####
            data_dict = self.forward_pass(image_batch)

            ##### Calculating losses and Back propagatiion #####
            loss_mse = self.mse_loss(data_dict['image_rec'], data_dict['image_ori'])
            loss_lpips = self.lpips_loss(data_dict['image_rec'], data_dict['image_ori'])

            SumTrainLosses = self.args.mse_lambda*loss_mse + self.args.lpips_lambda*loss_lpips

            self.encoder_optim.zero_grad()
            self.decoder_optim.zero_grad()
            SumTrainLosses.backward()
            self.encoder_optim.step()
            self.decoder_optim.step()
            
            ##### Log losses and computation time #####
            MSE_loss.update(loss_mse.item(), self.args.train_bs)
            LPIPS_loss.update(loss_lpips.item(), self.args.train_bs)
            TrainLosses.update(SumTrainLosses.item(), self.args.train_bs)

            batch_time.update(time.time()-start_time)
            start_time = time.time()

            train_data_dict = {
                'MSE_loss': MSE_loss.avg,
                'LPIPS_loss': LPIPS_loss.avg,
                'TrainLosses': TrainLosses.avg,
            }

            ##### Board and log losses, and visualize results #####
            if (self.global_train_steps+1) % self.args.board_interval == 0:
                train_log = "[{:d}/{:d}][Iteration: {:05d}][Steps: {:05d}] MSE_loss: {:.6f} LPIPS_loss: {:.6f} TrainLosses: {:.6f} BatchTime: {:.4f}".format(
                    epoch+1, self.args.max_epoch, train_iter+1, self.global_train_steps+1, MSE_loss.val, LPIPS_loss.val, TrainLosses.val, batch_time.val
                )
                print_log(info=train_log, log_path=self.args.logpath, console=True)
                log_metrics(writer=self.writer, data_dict=train_data_dict, step=self.global_train_steps+1, prefix='train')

            if self.global_train_steps == 0 or (self.global_train_steps+1) % self.args.image_interval == 0:
                visualize_psp_results(vis_dict=data_dict, dis_num=self.args.display_num, epoch=epoch+1, prefix='train', save_dir=self.args.trainpics_dir, iter=train_iter+1, step=self.global_train_steps+1)
            
            self.global_train_steps += 1
            
            if train_iter == self.args.max_train_iters-1:
                break


    def validation(self, epoch, image_loader):
        self.encoder.eval()
        self.decoder.eval()

        batch_time = AverageMeter()

        MSE_loss = AverageMeter()
        LPIPS_loss = AverageMeter()
        ValLosses = AverageMeter()

        start_time = time.time()

        for val_iter, image_batch in enumerate(image_loader):
            ##### Validation #####
            data_dict = self.forward_pass(image_batch)

            ##### Calculate losses #####
            loss_mse = self.mse_loss(data_dict['image_rec'], data_dict['image_ori'])
            loss_lpips = self.lpips_loss(data_dict['image_rec'], data_dict['image_ori'])

            SumValLosses = self.args.mse_lambda*loss_mse + self.args.lpips_lambda*loss_lpips

            ##### Log losses and computation time #####
            MSE_loss.update(loss_mse.item(), self.args.train_bs)
            LPIPS_loss.update(loss_lpips.item(), self.args.train_bs)
            ValLosses.update(SumValLosses.item(), self.args.train_bs)

            batch_time.update(time.time() - start_time)
            start_time = time.time()

            if val_iter == self.args.max_val_iters-1:
                break

        validate_data_dict = {
            'MSELoss': MSE_loss.avg,
            'LPIPS_loss': LPIPS_loss.avg,
            'ValLosses': ValLosses.avg,
        }
        log_metrics(writer=self.writer, data_dict=validate_data_dict, step=epoch+1, prefix='validate')

        val_log = "Validation[{:d}] MSE_loss: {:.6f} LPIPS_loss: {:.6f} Vallosses: {:.6f} BatchTime: {:.4f}".format(
            epoch+1, MSE_loss.avg, LPIPS_loss.avg, ValLosses.avg, batch_time.avg
        )
        print_log(info=val_log, log_path=self.args.logpath, console=True)

        visualize_psp_results(vis_dict=data_dict, dis_num=self.args.display_num, epoch=epoch+1, prefix='validation', save_dir=self.args.valpics_dir)

        return ValLosses.avg, data_dict


    def running(self):
        print_log("Training is beginning .......................................................", self.args.logpath)

        self.global_train_steps = 0

        for epoch in range(self.args.max_epoch):
            with torch.autograd.detect_anomaly(check_nan=False):
                self.training(epoch, self.train_loader)
            
            if epoch % self.args.validation_interval == 0:
                with torch.no_grad():
                    validation_loss, data_dict = self.validation(epoch, self.val_loader)

                self.encoder_scheduler.step()
                self.decoder_scheduler.step()
                
                stat_dict = {
                    'epoch': epoch + 1,
                    'encoder_state_dict': self.encoder.state_dict(),
                    'decoder_state_dict': self.decoder.state_dict(),
                }
                self.save_checkpoint(stat_dict, is_best=False)

                if (self.best_loss is None) or (validation_loss < self.best_loss):
                    self.best_loss = validation_loss
                    self.save_checkpoint(stat_dict, is_best=True)
                    
                    visualize_psp_results(vis_dict=data_dict, dis_num=self.args.display_num, epoch=epoch, prefix='best', save_dir=self.args.bestresults_dir)
        
        self.writer.close()
        print_log("Training finish .......................................................", self.args.logpath)


    def save_checkpoint(self, state, is_best):
        if is_best:
            torch.save(state['encoder_state_dict'], os.path.join(self.args.bestresults_dir, 'checkpoints', 'Encoder_best.pth'))
            torch.save(state['decoder_state_dict'], os.path.join(self.args.bestresults_dir, 'checkpoints', 'Decoder_best.pth'))
        else:
            torch.save(state, os.path.join(self.args.checkpoint_savedir, 'checkpoint.pth.tar'))



def main():
    args = TrainPspOptions().parse()

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