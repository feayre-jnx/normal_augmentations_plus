import torch
from torchvision import transforms

from datasets.APR import APRecombination
from datasets.AMP_ADJ import AMPAdjust
#from datasets.alb_augmentations import AlbumentationsApply
from datasets.cor_augmentations import NewAugmentationsApply


normalize = transforms.Compose([
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

def train_transforms(_transforms, input_size=32, alb_policy=None):
    transforms_list = []
    if 'aprs' in _transforms:
        print('APRecombination', _transforms)
        transforms_list.extend([
            transforms.Resize(input_size),   ## mosquito input 512 to 32 resize
            transforms.RandomApply([APRecombination(img_size=input_size)], p=1.0),
            transforms.RandomCrop(input_size, padding=4, fill=128),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    elif 'amp-adj' in _transforms:
        print('AMPAdjust', _transforms)
        transforms_list.extend([
            transforms.Resize(input_size),   ## mosquito input 512 to 32 resize
            transforms.RandomApply([AMPAdjust(img_size=input_size)], p=1.0),
            transforms.RandomCrop(input_size, padding=4, fill=128),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    elif 'album' in _transforms:
        print('Albumentations', _transforms)
        transforms_list.extend([
            transforms.Resize(input_size),   ## mosquito input 512 to 32 resize
            transforms.RandomApply([AlbumentationsApply(img_size=input_size, alb_policy=alb_policy)], p=1.0),
            transforms.RandomCrop(input_size, padding=4, fill=128),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            #normalize,  ## is this really needed?
        ])
    elif 'new-aug' in _transforms:
        print('NewAugmentations', _transforms)
        transforms_list.extend([
            transforms.Resize(input_size),   ## mosquito input 512 to 32 resize
            transforms.RandomApply([NewAugmentationsApply(img_size=input_size)], p=1.0),
            transforms.RandomCrop(input_size, padding=4, fill=128),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    else:
        transforms_list.extend([
            transforms.Resize(input_size),    ## mosquito input 512 to 32 resize
            transforms.RandomCrop(input_size, padding=4, fill=128),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

    return transforms_list



def test_transforms(input_size=32):
    test_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        ##normalize  ## placed here because of album
    ])  

    return test_transform