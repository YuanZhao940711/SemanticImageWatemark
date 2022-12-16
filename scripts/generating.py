import os
import sys
import numpy as np
from tqdm import tqdm

sys.path.append(".")
sys.path.append("..")

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from options.options import GenerateOptions

from network.AAD import AADGenerator
from network.MAE import MLAttrEncoder
from network.Fuser import Fuser
from network.Encoder import Encoder

from face_modules.model import Backbone

from utils.dataset import ImageDataset
from utils.common import tensor2img, alignment, l2_norm



class Generate:
    def __init__(self, args) -> None:
        
        self.args = args

        torch.backends.deterministic = True
        torch.backends.cudnn.benchmark = True
        SEED = self.args.seed
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

        self.args.device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
        print('[*]Running on device: {}'.format(self.args.device))

        ##### Initialize networks and load pretrained models #####
        assert self.args.checkpoint_dir != None, "[*]Please assign the right directory of the pre-trained models"
        print('[*]Loading pre-trained model from {}'.format(self.args.checkpoint_dir))
        
        # AAD Generator
        self.aadblocks = AADGenerator(c_id=512).to(self.args.device)
        self.aadblocks.load_state_dict(torch.load(os.path.join(self.args.checkpoint_dir, 'AAD_best.pth'), map_location=self.args.device), strict=True)
        # Att Encoder
        self.attencoder = MLAttrEncoder().to(self.args.device)
        self.attencoder.load_state_dict(torch.load(os.path.join(self.args.checkpoint_dir, 'ATT_best.pth'), map_location=self.args.device), strict=True)

        # Id Encoder
        print("[*]Loading Face Recognition Model {} from {}".format(self.args.facenet_mode, self.args.facenet_dir))
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
        self.fuser.load_state_dict(torch.load(os.path.join(self.args.checkpoint_dir, 'Fuser_best.pth'), map_location=self.args.device), strict=True)
        
        # Encoder
        self.encoder = Encoder(in_channels=3, latent_dim=self.args.latent_dim).to(self.args.device)
        self.encoder.load_state_dict(torch.load(os.path.join(self.args.checkpoint_dir, 'Encoder_best.pth'), map_location=self.args.device), strict=True)

        ##### Initialize data loaders ##### 
        cover_transforms = transforms.Compose([
            transforms.Resize([self.args.image_size, self.args.image_size]),
            transforms.ToTensor()
        ])
        secret_transforms = transforms.Compose([
            transforms.Resize([self.args.image_size, self.args.image_size]),
            transforms.ToTensor()
        ])

        self.cover_dataset = ImageDataset(root=self.args.cover_dir, transforms=cover_transforms)
        self.secret_dataset = ImageDataset(root=self.args.secret_dir, transforms=secret_transforms)
        print("[*]Loaded {} cover images".format(len(self.cover_dataset)))

        self.cover_loader = DataLoader(
            self.cover_dataset,
            batch_size=self.args.generate_bs,
            shuffle=False,
            num_workers=int(self.args.num_workers),
            drop_last=False
        )
        self.secret_loader = DataLoader(
            self.secret_dataset,
            batch_size=self.args.secret_bs,
            shuffle=False,
            num_workers=int(self.args.num_workers),
            drop_last=False
        )

    def running(self):
        print("Generation is beginning .......................................................")
        idx = 0
        
        secret_iterator = iter(self.secret_loader)
        for cover_batch in tqdm(self.cover_loader):
            try:
                secret_batch = next(secret_iterator)
            except StopIteration:
                secret_iterator = iter(self.secret_loader)
                secret_batch =next(secret_iterator)
            
            cover_batch = cover_batch.to(self.args.device)
            
            secret_batch = secret_batch.to(self.args.device)
            secret_batch = secret_batch.repeat(cover_batch.shape[0], 1, 1, 1)

            cover_att = self.attencoder(Xt=cover_batch)

            cover_id = self.facenet(alignment(cover_batch))
            cover_id_norm = l2_norm(cover_id)

            secret_feature = self.encoder(secret_batch)
            secret_feature_norm = l2_norm(secret_feature)

            fused_feature = self.fuser(cover_id_norm, secret_feature_norm)

            container_batch = self.aadblocks(inputs=(cover_att, fused_feature))

            for i, container in enumerate(container_batch):
                cover = tensor2img(cover_batch[i])
                secret = tensor2img(secret_batch[i])
                container = tensor2img(container)

                image_name = os.path.basename(self.cover_dataset.image_paths[idx]).split('.')[0]

                cover.save(os.path.join(self.args.cover_savedir, '{}.png'.format(image_name)))
                secret.save(os.path.join(self.args.secret_savedir, '{}.png'.format(image_name)))
                container.save(os.path.join(self.args.container_savedir, '{}.png'.format(image_name)))

                idx += 1



def main():
    args = GenerateOptions().parse()

    print('[*]Export generation results at {}'.format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)

    args.cover_savedir = os.path.join(args.output_dir, 'cover')
    os.makedirs(args.cover_savedir, exist_ok=True)

    args.secret_savedir = os.path.join(args.output_dir, 'secret_ori')
    os.makedirs(args.secret_savedir, exist_ok=True)

    args.container_savedir = os.path.join(args.output_dir, 'container')
    os.makedirs(args.container_savedir, exist_ok=True)

    generate = Generate(args)
    with torch.no_grad():
        generate.running()



if __name__ == '__main__':
    main()