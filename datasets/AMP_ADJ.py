import io
import random
from PIL import Image
import numpy as np
import datasets.augmentations as augmentations

class AMPAdjust(object):
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

        #p = random.uniform(0, 1)
        #if p > 0.5:
        #    op = np.random.choice(self.aug_list)
        #    x = op(x, 3)

        #op = np.random.choice(self.aug_list)
        #x = op(x, 3)

        p = random.uniform(0, 1)
        if p > 0.5:
            return x

        x = np.array(x).astype(np.uint8) 
        
        fft_1 = np.fft.fftshift(np.fft.fftn(x))
        abs_1, angle_1 = np.abs(fft_1), np.angle(fft_1)

        ## divide the amplitude (abs)
        divider = np.random.choice([.7, .5, .3, .1])
        abs_1 = abs_1 * divider

        fft_1 = abs_1*np.exp((1j) * angle_1)

        x = x.astype(np.uint8)
        x = Image.fromarray(x)
        
        return x
