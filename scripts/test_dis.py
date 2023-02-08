import os
import sys
import numpy as np
from tqdm import tqdm

sys.path.append(".")
sys.path.append("..")

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from options.options import TestDisOptions

from network.DisentanglementEncoder import DisentanglementEncoder
from network.AAD import AADGenerator

from utils.dataset import ImageDataset
from utils.common import visualize_distest_results, l2_norm




class Test:
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
        print("Running Dis testing .......................................................")
        idx = 0

        for image in tqdm(self.image_loader):
            image = image.to(self.args.device)

            image_ori = image.repeat(8,1,1,1)

            image_ori_id, image_ori_att = self.disentangler(image_ori)
            image_ori_id_norm = l2_norm(image_ori_id)

            image_rec = self.generator(inputs=(image_ori_att, image_ori_id_norm))

            image_rec_id, _ = self.disentangler(image_rec)
            image_rec_id_norm = l2_norm(image_rec_id)

            img_name = os.path.basename(self.image_dataset.image_paths[idx]).split('.')[0]

            data_dict = {
                'image_ori': image_ori,
                'image_rec': image_rec,
                'image_ori_id': image_ori_id_norm,
                'image_rec_id': image_rec_id_norm,
            }

            visualize_distest_results(vis_dict=data_dict, save_dir=os.path.join(self.args.output_dir, '{}.png'.format(img_name)))

            idx += 1



def main():
    args = TestDisOptions().parse()

    print('[*]Export test results at {}'.format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)

    test = Test(args)
    with torch.no_grad():
        test.running()



if __name__ == '__main__':
	main()