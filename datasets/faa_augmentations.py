import datasets.augmentations as augmentations
from .faster_autoaugment.policy import Policy
import torchvision.transforms as transforms
import torch

class FAAAugmentationsApply(object):
    def __init__(self, img_size=32, aug=None):
        if aug is None:
            augmentations.IMAGE_SIZE = img_size     ## Does not take into effect, change the value directly from the function inside augmentations.py
            self.aug_list = augmentations.augmentations
        else:
            self.aug_list = aug.augmentations

        path = 'datasets/faster_autoaugment/policy_weights/19_mq_hq.pt'
        num_chunks = 16

        policy_weight = torch.load(path, map_location='cpu')
        self.policy = Policy.faster_auto_augment_policy(num_chunks=num_chunks, **policy_weight['policy_kwargs'])
        self.policy.load_state_dict(policy_weight['policy'])

    def __call__(self, x):
        '''
        :param img: (PIL Image): Image
        :return: code img (PIL Image): Image
        '''
        
        ## FAA augmentation
        x = transforms.ToTensor()(x).unsqueeze(0)
        x_aug = self.policy(self.policy.denormalize_(x))

        output = transforms.ToPILImage()(x_aug[0].squeeze(0))

        return output
