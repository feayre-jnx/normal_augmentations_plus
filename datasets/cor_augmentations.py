import io
import random
from PIL import Image
import numpy as np
import datasets.augmentations as augmentations
from common_corruption.imagecorruptions import corrupt

class NewAugmentationsApply(object):
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
        
        ## new (downsampling) augmentations

        p = 0#random.uniform(0, 1)
        if p > 0.5:   
            w, h = x.size
            ds_rate = np.random.choice([2, 3, 4, 5, 6])
            downsamp_x = x.resize(size=(int(w/ds_rate), int(h/ds_rate)), resample=Image.BICUBIC)
            upsamp_x = downsamp_x.resize(size=(w, h), resample=Image.BICUBIC)
            x = upsamp_x

        ## new (corruptions) augmentations

        p = random.uniform(0, 1)
        if p > 0.5:
            corruption_choice = np.random.choice(15)
            severity = np.random.choice(5, p=[.15,.22,.26,.22,.15]) + 1
            x = np.array(x).astype(np.uint8) 
            corrupted_image = corrupt(x, corruption_number=corruption_choice, severity=severity)
            x = corrupted_image
            x = Image.fromarray(x)

        ## normal augmentations

        p = 0#random.uniform(0, 1)
        if p > 0.5:
            return x

        op = np.random.choice(self.aug_list)
        x = op(x, 3)
        
        return x
