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

from options.options import TrainDisOptions

from network.DisentanglementEncoder import DisentanglementEncoder
from network.AAD import AADGenerator

from criteria import loss_functions

from utils.dataset import ImageDataset
from utils.common import weights_init, visualize_dis_results, print_log, log_metrics, l2_norm, AverageMeter




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
            print_log("[*]Training Disentangle Encoder from scratch", self.args.logpath)
            self.disentangler.apply(weights_init)

        # AAD Generator
        self.generator= AADGenerator(c_id=512).to(self.args.device)
        try:
            self.generator.load_state_dict(torch.load(os.path.join(self.args.checkpoint_dir, 'Gen_best.pth'), map_location=self.args.device), strict=True)
            print_log("[*]Successfully loaded Generator's pre-trained model", self.args.logpath)
        except:
            print_log("[*]Training AAD Generator from scratch", self.args.logpath)
            self.generator.apply(weights_init)


        ##### Initialize optimizers #####
        self.dis_optim = optim.Adam(self.disentangler.parameters(), lr=self.args.dis_lr, betas=(0.5, 0.999))
        self.gen_optim = optim.Adam(self.generator.parameters(), lr=self.args.gen_lr, betas=(0.5, 0.999))

        #self.dis_scheduler = optim.lr_scheduler.ExponentialLR(optimizer=self.dis_optim, gamma=0.9, last_epoch=-1, verbose=True)
        #self.gen_scheduler = optim.lr_scheduler.ExponentialLR(optimizer=self.gen_optim, gamma=0.9, last_epoch=-1, verbose=True)

        ##### Initialize loss functions #####
        self.att_loss = loss_functions.AttLoss().to(self.args.device).eval()
        self.id_loss = loss_functions.IdLoss(self.args.idloss_mode).to(self.args.device).eval()
        self.rec_loss = loss_functions.RecConLoss(self.args.recloss_mode, self.args.device).eval()


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
        print_log(info="[*]Loaded {} training cover images".format(len(train_dataset)), log_path=self.args.logpath, console=True)
        print_log(info="[*]Loaded {} validation cover images".format(len(val_dataset)), log_path=self.args.logpath, console=True)

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

        image_ori_id, image_ori_att = self.disentangler(image_ori)

        image_rec = self.generator(inputs=(image_ori_att, image_ori_id))

        image_rec_id, image_rec_att = self.disentangler(image_rec)

        ##### Collect results ##### 
        data_dict = {
            'image_ori': image_ori,
            'image_rec': image_rec,
            'image_ori_id': image_ori_id,
            'image_ori_att': image_ori_att,
            'image_rec_id': image_rec_id,
            'image_rec_att': image_rec_att,
        }

        return data_dict


    def training(self, epoch, image_loader):
        self.disentangler.train()
        self.generator.train()

        batch_time = AverageMeter()

        Att_loss = AverageMeter()
        Id_loss = AverageMeter()
        Rec_loss = AverageMeter()

        Train_losses = AverageMeter()

        start_time = time.time()

        for train_iter, image_batch in enumerate(image_loader):
            ##### Training #####
            data_dict = self.forward_pass(image_batch)

            loss_att = self.att_loss(data_dict['image_rec_att'], data_dict['image_ori_att'])
            loss_id = self.id_loss(data_dict['image_rec_id'], data_dict['image_ori_id'])
            loss_rec = self.rec_loss(data_dict['image_rec'], data_dict['image_ori'])

            Sum_train_losses = self.args.att_lambda*loss_att + self.args.id_lambda*loss_id + self.args.rec_lambda*loss_rec

            self.dis_optim.zero_grad()
            self.gen_optim.zero_grad()

            Sum_train_losses.backward()

            self.dis_optim.step()
            self.gen_optim.step()
            
            ##### Log losses and computation time #####
            Att_loss.update(loss_att.item(), self.args.train_bs)
            Id_loss.update(loss_id.item(), self.args.train_bs)
            Rec_loss.update(loss_rec.item(), self.args.train_bs)
            Train_losses.update(Sum_train_losses.item(), self.args.train_bs)

            batch_time.update(time.time()-start_time)
            start_time = time.time()

            train_data_dict = {
                'AttLoss': Att_loss.avg,
                'IdLoss': Id_loss.avg,
                'RecLoss': Rec_loss.avg,
                'SumTrainLosses': Train_losses.avg,
            }

            ##### Board and log losses, and visualize results #####
            if (self.global_train_steps+1) % self.args.board_interval == 0:
                train_log = "[{:d}/{:d}][Iteration: {:05d}][Steps: {:05d}] Att_loss: {:.6f} Id_loss: {:.6f} Rec_loss: {:.6f} Sumlosses={:.6f} BatchTime: {:.4f}".format(
                    epoch+1, self.args.max_epoch, train_iter+1, self.global_train_steps+1, Att_loss.val, Id_loss.val, Rec_loss.val, Train_losses.val, batch_time.val
                )
                print_log(info=train_log, log_path=self.args.logpath, console=True)
                log_metrics(writer=self.writer, data_dict=train_data_dict, step=self.global_train_steps+1, prefix='train')

            if self.global_train_steps == 0 or (self.global_train_steps+1) % self.args.image_interval == 0:
                visualize_dis_results(vis_dict=data_dict, dis_num=self.args.display_num, epoch=epoch+1, prefix='train', save_dir=self.args.trainpics_dir, iter=train_iter+1, step=self.global_train_steps+1)
            
            self.global_train_steps += 1
            
            if train_iter == self.args.max_train_iters-1:
                break


    def validation(self, epoch, image_loader):
        self.disentangler.eval()
        self.generator.eval()

        batch_time = AverageMeter()

        Att_loss = AverageMeter()
        Id_loss = AverageMeter()
        Rec_loss = AverageMeter()
        Val_losses = AverageMeter()

        start_time = time.time()

        for val_iter, image_batch in enumerate(image_loader):
            ##### Validation #####
            data_dict = self.forward_pass(image_batch)

            # Calculate losses
            loss_att = self.att_loss(data_dict['image_rec_att'], data_dict['image_ori_att'])
            loss_id = self.id_loss(data_dict['image_rec_id'], data_dict['image_ori_id'])
            loss_rec = self.rec_loss(data_dict['image_rec'], data_dict['image_ori'])

            sum_val_loss = self.args.att_lambda*loss_att + self.args.id_lambda*loss_id + self.args.rec_lambda*loss_rec

            ##### Log losses and computation time #####
            Att_loss.update(loss_att.item(), self.args.train_bs)
            Id_loss.update(loss_id.item(), self.args.train_bs)
            Rec_loss.update(loss_rec.item(), self.args.train_bs)
            Val_losses.update(sum_val_loss.item(), self.args.train_bs)

            batch_time.update(time.time() - start_time)
            start_time = time.time()

            if val_iter == self.args.max_val_iters-1:
                break

        validate_data_dict = {
            'AttLoss': Att_loss.avg,
            'IdLoss': Id_loss.avg,
            'RecLoss': Rec_loss.avg,
            'SumValidateLosses': Val_losses.avg,
        }
        log_metrics(writer=self.writer, data_dict=validate_data_dict, step=epoch+1, prefix='validate')

        val_log = "Validation[{:d}] Att_loss: {:.6f} Id_loss: {:.6f} Rec_loss: {:.6f} Sumlosses={:.6f} BatchTime: {:.4f}".format(
            epoch+1, Att_loss.avg, Id_loss.avg, Rec_loss.avg, Val_losses.avg, batch_time.avg
        )
        print_log(info=val_log, log_path=self.args.logpath, console=True)

        visualize_dis_results(vis_dict=data_dict, dis_num=self.args.display_num, epoch=epoch+1, prefix='validation', save_dir=self.args.valpics_dir)

        return Val_losses.avg, data_dict


    def running(self):
        print_log("Training is beginning .......................................................", self.args.logpath)

        self.global_train_steps = 0

        for epoch in range(self.args.max_epoch):
            self.training(epoch, self.train_loader)
            
            if (epoch+1) % self.args.validation_interval == 0:
                with torch.no_grad():
                    validation_loss, data_dict = self.validation(epoch, self.val_loader)
                
                #self.dis_scheduler.step()
                #self.gen_scheduler.step()
                
                stat_dict = {
                    'epoch': epoch + 1,
                    'dis_state_dict': self.disentangler.state_dict(),
                    'gen_state_dict': self.generator.state_dict(),
                }
                self.save_checkpoint(stat_dict, is_best=False)

                if (self.best_loss is None) or (validation_loss < self.best_loss):
                    self.best_loss = validation_loss
                    self.save_checkpoint(stat_dict, is_best=True)

                    visualize_dis_results(vis_dict=data_dict, dis_num=self.args.display_num, epoch=epoch, prefix='best', save_dir=self.args.bestresults_dir)

        self.writer.close()
        print_log("Training finish .......................................................", self.args.logpath)


    def save_checkpoint(self, state, is_best):
        if is_best:
            torch.save(state['dis_state_dict'], os.path.join(self.args.bestresults_dir, 'checkpoints', 'Dis_best.pth'))
            torch.save(state['gen_state_dict'], os.path.join(self.args.bestresults_dir, 'checkpoints', 'Gen_best.pth'))
        else:
            torch.save(state, os.path.join(self.args.checkpoint_savedir, 'checkpoint.pth.tar'))



def main():
    args = TrainDisOptions().parse()

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