import os
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms



IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
]
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    image_paths = []
    assert os.path.isdir(dir), '[*]{} is not a valid directory'.format(dir)
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root,fname)
                image_paths.append(path)                
    assert len(image_paths) > 0, '[*]The number of input images should not zero'
    return image_paths
    
def select_dataset(dir, max_num, rand_select, rand_seed):
    image_paths = []
    assert os.path.isdir(dir), '[*]{} is not a valid directory'.format(dir)
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root,fname)
                image_paths.append(path)
    print("[*]Loaded {} original images, selected {} images for watermarking".format(len(image_paths), max_num))
    assert len(image_paths) >= max_num, '[*]Total loaded images number should bigger than selected images'
    if rand_select=='Yes':
        np.random.seed(rand_seed)
        selected_imgpaths = np.random.choice(image_paths, max_num)
    else:
        selected_imgpaths = image_paths[:max_num]
    return selected_imgpaths

def label_dataset(dir, label):
    data_paths = []
    assert os.path.isdir(dir), '[*]{} is not a valid directory'.format(dir)
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root,fname)
                data_paths.append((path, label))                
    assert len(data_paths) > 0, '[*]The number of input images should not zero'
    return data_paths



class ImageDataset(Dataset):
    def __init__(self, root, transforms):
        self.image_paths = sorted(make_dataset(root))
        self.transforms = transforms
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, index):
        image_path = self.image_paths[index]

        try:
            #image_path = self.image_paths[index]
            #image = Image.open(image_path).convert("RGB")
            #return self.transforms(image)
            image = Image.open(image_path).convert('RGB')
            image = self.transforms(image)
            
            return image
        except:
            self.__getitem__(index + 1)



class TrainDataset(Dataset):
    def __init__(self, root):
        print("[*]Loading Images from {}".format(root))
        self.image_paths = sorted(make_dataset(root))
        print('[*]Totally loaded {} images'.format(len(self.image_paths)))
        self.transforms = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor()
        ])
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        try:
            image = Image.open(image_path)
        except:
            while True:
                image_path = random.choice(self.image_paths)
                try:
                    image = Image.open(image_path)
                except:
                    continue
                break
        image = image.convert('RGB')
        return self.transforms(image)



class InjectDataset(Dataset):
    def __init__(self, root, max_num, rand_select, rand_seed):
        print("[*]Loading Images from {}".format(root))
        self.image_paths = sorted(select_dataset(root, max_num, rand_select, rand_seed))
        print('[*]Totally loaded {} images'.format(len(self.image_paths)))
        self.transforms = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor()
        ])
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path)
        image = image.convert('RGB')
        return self.transforms(image)



class AnalyseDataset(Dataset):
    def __init__(self, root, perturbation):
        print("[*]Loading Images from {}".format(root))
        self.image_paths = sorted(make_dataset(root))
        print('[*]Totally loaded {} images'.format(len(self.image_paths)))
        if perturbation == 'Yes':
            print("[*]Loading image with perturbation")
            self.transforms = transforms.Compose([
                transforms.RandomResizedCrop(size=256, scale=(0.6, 1.0)),
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.01)
            ])
        else:
            print("[*]Loading image without perturbation")
            self.transforms = transforms.Compose([
                transforms.Resize((256,256)),
                transforms.ToTensor()
            ])
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path)
        image = image.convert('RGB')
        return self.transforms(image)



class EvaluateDataset(Dataset):
    def __init__(self, pos_root, neg_root):
        self.data_list = []

        print("[*]Loading positive images from {}".format(pos_root))
        pos_paths = sorted(label_dataset(pos_root, label=1))
        self.data_list.extend(pos_paths)

        print("[*]Loading negative images from {}".format(neg_root))
        neg_paths = sorted(label_dataset(neg_root, label=0))
        self.data_list.extend(neg_paths)

        print('[*]Totally loaded {} images, with {} positive images and {} negative images'.format(len(self.data_list), len(pos_paths), len(neg_paths)))

        self.transforms = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor()
        ])
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, index):
        image_path, label = self.data_list[index]
        image = Image.open(image_path)
        image = image.convert('RGB')
        return self.transforms(image), label