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

from face_modules.model import Backbone
from network.MAE import MLAttrEncoder
from network.AAD import AADGenerator
from network.Fuser import Fuser
from network.Encoder import PspEncoder

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

        # Id Encoder
        self.idencoder = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se').to(self.args.device)
        self.idencoder.load_state_dict(torch.load(os.path.join(self.args.checkpoint_dir, 'Id_best.pth'), map_location=self.args.device), strict=True)
        #self.idencoder.load_state_dict(torch.load(self.args.checkpoint_dir, map_location=self.args.device)['id_state_dict'], strict=True)
        
        # Att Encoder
        self.attencoder = MLAttrEncoder().to(self.args.device)
        self.attencoder.load_state_dict(torch.load(os.path.join(self.args.checkpoint_dir, 'Att_best.pth'), map_location=self.args.device), strict=True)
        #self.attencoder.load_state_dict(torch.load(self.args.checkpoint_dir, map_location=self.args.device)['att_state_dict'], strict=True)

        # AAD Generator
        self.generator = AADGenerator(c_id=512).to(self.args.device)
        self.generator.load_state_dict(torch.load(os.path.join(self.args.checkpoint_dir, 'Gen_best.pth'), map_location=self.args.device), strict=True)
        #self.generator.load_state_dict(torch.load(self.args.checkpoint_dir, map_location=self.args.device)['gen_state_dict'], strict=True)

        # Fuser
        self.fuser = Fuser(latent_dim=self.args.latent_dim).to(self.args.device)
        self.fuser.load_state_dict(torch.load(os.path.join(self.args.checkpoint_dir, 'Fuser_best.pth'), map_location=self.args.device), strict=True)
        #self.fuser.load_state_dict(torch.load(self.args.checkpoint_dir, map_location=self.args.device)['fuser_state_dict'], strict=True)

        # Encoder
        self.encoder = PspEncoder(num_layers=50, mode='ir_se').to(self.args.device)
        self.encoder.load_state_dict(torch.load(os.path.join(self.args.checkpoint_dir, 'Encoder_best.pth'), map_location=self.args.device), strict=True)
        #self.encoder.load_state_dict(torch.load(self.args.checkpoint_dir, map_location=self.args.device)['encoder_state_dict'], strict=True)
        
        #"""
        self.idencoder.eval()
        self.attencoder.eval()
        self.generator.eval()
        self.fuser.eval()
        self.encoder.eval()
        #"""

        """
        self.idencoder.train()
        self.attencoder.train()
        self.generator.train()
        self.fuser.train()
        self.encoder.train()
        """

        ##### Initialize data loaders ##### 
        cover_transforms = transforms.Compose([
            transforms.Resize([self.args.image_size, self.args.image_size]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ])
        secret_transforms = transforms.Compose([
            transforms.Resize([self.args.secret_size, self.args.secret_size]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ])

        self.cover_dataset = ImageDataset(root=self.args.cover_dir, transforms=cover_transforms)
        self.secret_dataset = ImageDataset(root=self.args.secret_dir, transforms=secret_transforms)
        print("[*]Loaded {} cover images".format(len(self.cover_dataset)))
        print("[*]Loaded {} secret images".format(len(self.secret_dataset)))

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

            cover_att = self.attencoder(cover_batch)
            cover_id = self.idencoder(alignment(cover_batch))
            cover_id_norm = l2_norm(cover_id)

            secret_feature_ori = self.encoder(secret_batch)
            secret_feature_ori_norm = l2_norm(secret_feature_ori)

            input_feature = self.fuser(cover_id=cover_id_norm, secret_feat=secret_feature_ori_norm)
            input_feature_norm = l2_norm(input_feature)

            #container_batch = self.generator(inputs=(cover_att, input_feature))
            container_batch = self.generator(inputs=(cover_att, input_feature_norm))

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