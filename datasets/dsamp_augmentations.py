import io
import random
from PIL import Image
import numpy as np
import datasets.augmentations as augmentations
from common_corruption.imagecorruptions import corrupt

class DSampAugmentationsApply(object):
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
        
        ## downsampling augmentation
        w, h = x.size
        ds_rate = np.random.choice([2, 3, 4, 5, 6])
        downsamp_x = x.resize(size=(int(w/ds_rate), int(h/ds_rate)), resample=Image.BICUBIC)
        upsamp_x = downsamp_x.resize(size=(w, h), resample=Image.BICUBIC)
        x = upsamp_x

        
        return x
