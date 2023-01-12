import os
import sys
import numpy as np
from tqdm import tqdm

sys.path.append(".")
sys.path.append("..")

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from options.options import ExtractOptions

from network.DisentanglementEncoder import DisentanglementEncoder
from network.Separator import Separator
from stylegan2.model import Generator

from utils.dataset import ImageDataset
from utils.common import tensor2img, alignment



class Extract:
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

        # Disentanglement Encoder
        self.disentangler = DisentanglementEncoder(latent_dim=self.args.latent_dim).to(self.args.device)
        self.disentangler.load_state_dict(torch.load(os.path.join(self.args.checkpoint_dir, 'Dis_best.pth'), map_location=self.args.device), strict=True)
        
        # Separator
        self.separator = Separator(latent_dim=self.args.latent_dim).to(self.args.device)
        self.separator.load_state_dict(torch.load(os.path.join(self.args.checkpoint_dir, 'Separator_best.pth'), map_location=self.args.device), strict=True)

        # Decoder
        self.decoder = Generator(latent_dim=self.args.latent_dim).to(self.args.device)
        self.decoder.load_state_dict(torch.load(os.path.join(self.args.checkpoint_dir, 'Decoder_best.pth'), map_location=self.args.device), strict=True)

        self.disentangler.eval()
        self.separator.eval()
        self.decoder.eval()        
        
        ##### Initialize data loaders ##### 
        container_transforms = transforms.Compose([
            transforms.Resize([self.args.image_size, self.args.image_size]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ])

        self.container_dataset = ImageDataset(root=self.args.container_dir, transforms=container_transforms)
        print("[*]Loaded {} container images".format(len(self.container_dataset)))

        self.container_loader = DataLoader(
            self.container_dataset,
            batch_size=self.args.extract_bs,
            shuffle=False,
            num_workers=int(self.args.num_workers),
            drop_last=False
        )

    def running(self):
        print("Extraction is beginning .......................................................")
        idx = 0
        
        for container_batch in tqdm(self.container_loader):

            container_batch = container_batch.to(self.args.device)

            container_id, _ = self.disentangler(container_batch)

            secret_feature_rec = self.separator(container_id)
            #secret_feature_rec = l2_norm(secret_feature_rec)

            secret_rec_batch, _ = self.decoder(
                styles=[secret_feature_rec],
                input_is_latent=True,
                randomize_noise=True,
                return_latents=False,
            )

            for secret_rec in secret_rec_batch:
                secret_rec = tensor2img(secret_rec)

                image_name = os.path.basename(self.container_dataset.image_paths[idx]).split('.')[0]

                secret_rec.save(os.path.join(self.args.rec_secret_savedir, '{}.png'.format(image_name)))

                idx += 1



def main():
    args = ExtractOptions().parse()

    print('[*]Export generation results at {}'.format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)

    args.rec_secret_savedir = os.path.join(args.output_dir, 'secret_rec')
    os.makedirs(args.rec_secret_savedir, exist_ok=True)

    extract = Extract(args)
    with torch.no_grad():
        extract.running()



if __name__ == '__main__':
    main()