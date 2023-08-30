from PIL import Image
import numpy as np
import datasets.augmentations as augmentations
from .faster_autoaugment.policy_mq import Policy
import torchvision.transforms as transforms
import torch

class FAAAugmentationsApply(object):
    def __init__(self, img_size=32, aug=None):
        if aug is None:
            augmentations.IMAGE_SIZE = img_size     ## Does not have any meaning, change the value directly from the function
            self.aug_list = augmentations.augmentations
        else:
            self.aug_list = aug.augmentations

    def __call__(self, x):
        '''
        :param img: (PIL Image): Image
        :return: code img (PIL Image): Image
        '''
        
        ## FAA augmentation
        path = 'datasets/faster_autoaugment/policy_weights/19_mq4.pt'
        num_chunks = 16

        x = transforms.ToTensor()(x).unsqueeze(0)
        policy_weight = torch.load(path, map_location='cpu')
        policy = Policy.faster_auto_augment_policy(num_chunks=num_chunks, **policy_weight['policy_kwargs'])
        policy.load_state_dict(policy_weight['policy'])

        x_aug = policy(policy.denormalize_(x))

        output = transforms.ToPILImage()(x_aug[0].squeeze(0))

        return output
