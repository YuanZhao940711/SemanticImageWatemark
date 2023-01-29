import os
import sys
import numpy as np
from tqdm import tqdm

sys.path.append(".")
sys.path.append("..")

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from options.options import InferencePspOptions

from network.Encoder import PspEncoder
from stylegan2.model import Stylegan2Decoder

from utils.dataset import ImageDataset
from utils.common import tensor2img, l2_norm




class Inference:
    def __init__(self, args):
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
        #checkpoint = torch.load(self.args.checkpoint_dir, map_location=self.args.device)

        # Encoder
        self.encoder = PspEncoder(50, 'ir_se').to(self.args.device)
        #self.encoder.load_state_dict(torch.load(os.path.join(self.args.checkpoint_dir, 'Encoder_best.pth'), map_location=self.args.device), strict=True)
        try:
            self.encoder.load_state_dict(torch.load(os.path.join(self.args.checkpoint_dir, 'Encoder_best.pth'), map_location=self.args.device), strict=True)
            #self.encoder.load_state_dict(checkpoint['encoder_state_dict'], strict=True)
            print("[*]Successfully loaded Encoder's pre-trained model")
            #torch.save(self.encoder.state_dict(), os.path.join(self.args.output_dir, 'Encoder_best.pth'))
        except:
            raise ValueError("[*]Unable to load Encoder's pre-trained model")

        # Decoder
        #default size=1024, style_dim=512, n_mlp=8
        self.decoder = Stylegan2Decoder(size=self.args.image_size, style_dim=self.args.latent_dim, n_mlp=8).to(self.args.device)
        try:
            self.decoder.load_state_dict(torch.load(os.path.join(self.args.checkpoint_dir, 'Decoder_best.pth'), map_location=self.args.device), strict=True)
            #self.decoder.load_state_dict(checkpoint['decoder_state_dict'], strict=True)
            print("[*]Successfully loaded Decoder's pre-trained model")
            #torch.save(self.decoder.state_dict(), os.path.join(self.args.output_dir, 'Decoder_best.pth'))
        except:
            raise ValueError("[*]Unable to load Decoder's pre-trained model")
        
        self.encoder.eval()
        self.decoder.eval()

        ##### Initialize data loaders ##### 
        image_transforms = transforms.Compose([
            transforms.Resize([self.args.image_size, self.args.image_size]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ])

        self.image_dataset = ImageDataset(root=self.args.image_dir, transforms=image_transforms)
        print("[*]Loaded {} images".format(len(self.image_dataset)))
        
        self.image_loader = DataLoader(
            self.image_dataset,
            batch_size=self.args.batchsize,
            shuffle=False,
            num_workers=int(self.args.num_workers),
            drop_last=False,
        )


    def running(self):
        print("Running Psp reconstruction .......................................................")
        idx = 0

        for image_batch in tqdm(self.image_loader):
            image_ori = image_batch.to(self.args.device)

            image_feature = self.encoder(image_ori)
            image_feature_norm = l2_norm(image_feature)

            image_rec, _ = self.decoder(
                #styles=[image_feature_plus],
                styles=[image_feature_norm],
                input_is_latent=True,
                randomize_noise=True,
                return_latents=False,
            )

            for img_ori, img_rec in zip(image_ori, image_rec):
                img_ori = tensor2img(img_ori)
                img_rec = tensor2img(img_rec)

                img_name = os.path.basename(self.image_dataset.image_paths[idx]).split('.')[0]

                img_ori.save(os.path.join(self.args.imgori_savedir, '{}.png'.format(img_name)))
                img_rec.save(os.path.join(self.args.imgrec_savedir, '{}.png'.format(img_name)))

                idx += 1



def main():
    args = InferencePspOptions().parse()

    print('[*]Export inference results at {}'.format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)

    args.imgori_savedir = os.path.join(args.output_dir, 'image_ori')
    os.makedirs(args.imgori_savedir, exist_ok=True)

    args.imgrec_savedir = os.path.join(args.output_dir, 'image_rec')
    os.makedirs(args.imgrec_savedir, exist_ok=True)

    inference = Inference(args)
    with torch.no_grad():
        inference.running()



if __name__ == '__main__':
	main()