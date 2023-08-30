import torch
from torchvision import transforms

from datasets.APR import APRecombination
from datasets.AMP_ADJ import AMPAdjust
#from datasets.alb_augmentations import AlbumentationsApply
from datasets.cor_augmentations import ComCorAugmentationsApply
from datasets.dsamp_augmentations import DSampAugmentationsApply
from datasets.faa_augmentations import FAAAugmentationsApply


normalize = transforms.Compose([
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def train_transforms(_transforms, input_size=32, alb_policy=None, aug_set=None):
    transforms_list = []

    ## opening transforms
    transforms_list.extend([
                transforms.Resize(input_size),   
            ])

    ## main transforms
    for i in _transforms: 

        if i[0] == 'aprs':
            print('APRecombination', i)
            transforms_list.extend([
                transforms.RandomApply([APRecombination(img_size=input_size, aug=aug_set)], p=i[1])
            ])
        elif i[0] == 'normal':
            print('NormalAugmentations', i)
            transforms_list.extend([
                transforms.RandomApply([APRecombination(img_size=input_size, aug=aug_set, just_aug=True)], p=i[1])
            ])
        elif i[0] == 'amp-adj':
            print('AMPAdjust', i)
            transforms_list.extend([
                transforms.RandomApply([AMPAdjust(img_size=input_size)], p=i[1])
            ])
        elif i[0] == 'com-cor':
            print('ComCorAugmentations', i)
            transforms_list.extend([
                transforms.RandomApply([ComCorAugmentationsApply(img_size=input_size)], p=i[1])
            ])
        elif i[0] == 'dsamp':
            print('DSampAugmentations', i)
            transforms_list.extend([
                transforms.RandomApply([DSampAugmentationsApply(img_size=input_size)], p=i[1])
            ])

        ## experimental extension
        elif i[0] == 'album':
            print('Albumentations', i)
            transforms_list.extend([
                transforms.RandomApply([AlbumentationsApply(img_size=input_size, alb_policy=alb_policy)], p=i[1])
            ])

        elif i[0] == 'faa':
            print('FasterAutoAugment', i)
            transforms_list.extend([
                transforms.RandomApply([FAAAugmentationsApply(img_size=input_size)], p=i[1])
            ])
    
    ## closing transforms
    transforms_list.extend([
                transforms.RandomCrop(input_size, padding=4, fill=128),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
    
    if len(_transforms) == 0:
        print('No Augmentations')

    return transforms_list



def test_transforms(input_size=32):
    test_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
    ])  

    return test_transform