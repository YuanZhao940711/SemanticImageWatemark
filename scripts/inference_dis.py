import os
import sys
import numpy as np
from tqdm import tqdm

sys.path.append(".")
sys.path.append("..")

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from options.options import InferenceDisOptions

from network.DisentanglementEncoder import DisentanglementEncoder
from network.AAD import AADGenerator

from utils.dataset import ImageDataset
from utils.common import tensor2img




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
        # Disentanglement Encoder
        self.disentangler = DisentanglementEncoder(latent_dim=self.args.latent_dim).to(self.args.device)
        try:
            self.disentangler.load_state_dict(torch.load(os.path.join(self.args.checkpoint_dir, 'Dis_best.pth'), map_location=self.args.device), strict=True)
            print("[*]Successfully loaded Disentangler's pre-trained model")
        except:
            raise ValueError("[*]Unable to load Disentangler's pre-trained model")

        # AAD Generator
        self.generator= AADGenerator(c_id=512).to(self.args.device)
        try:
            self.generator.load_state_dict(torch.load(os.path.join(self.args.checkpoint_dir, 'Gen_best.pth'), map_location=self.args.device), strict=True)
            print("[*]Successfully loaded Generator's pre-trained model")
        except:
            raise ValueError("[*]Unable to load Generator's pre-trained model")

        self.disentangler.eval()
        self.generator.eval()


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
        print("Running Dis reconstruction .......................................................")
        idx = 0

        for image_batch in tqdm(self.image_loader):
            image_ori = image_batch.to(self.args.device)

            image_id, image_att = self.disentangler(image_ori)
            
            image_rec = self.generator(inputs=(image_att, image_id))

            for img_ori, img_rec in zip(image_ori, image_rec):
                img_ori = tensor2img(img_ori)
                img_rec = tensor2img(img_rec)

                img_name = os.path.basename(self.image_dataset.image_paths[idx]).split('.')[0]

                img_ori.save(os.path.join(self.args.imgori_savedir, '{}.png'.format(img_name)))
                img_rec.save(os.path.join(self.args.imgrec_savedir, '{}.png'.format(img_name)))

                idx += 1



def main():
    args = InferenceDisOptions().parse()

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