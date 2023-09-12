import random
from PIL import Image
import numpy as np
import datasets.augmentations as augmentations

class APRecombination(object):
    def __init__(self, img_size=32, aug=None, just_aug=False):
        if aug is None:
            augmentations.IMAGE_SIZE = img_size     ## Does not take into effect, change the value directly from the function inside augmentations.py
            self.aug_list = augmentations.augmentations

        elif aug == 'geo-photo':
            self.aug_list = augmentations.augmentations_pil_ver_geo_photo

        elif aug == 'geo-photo-k':
            self.aug_list = augmentations.augmentations_kornia_ver_geo_photo
        
        elif aug == 'geo':
            self.aug_list = augmentations.augmentations_pil_ver_geo

        elif aug == 'photo':
            self.aug_list = augmentations.augmentations_pil_ver_photo

        elif aug == 'geo-k':
            self.aug_list = augmentations.augmentations_kornia_ver_geo

        elif aug == 'photo-k':
            self.aug_list = augmentations.augmentations_kornia_ver_photo

        else:
            self.aug_list = aug.augmentations
        
        self.just_aug = just_aug

    def __call__(self, x):
        '''
        :param img: (PIL Image): Image
        :return: code img (PIL Image): Image
        '''
        
        ## the options['main_aug'] = normal (i.e., no aprs, just augmentations)
        if self.just_aug:
            op = np.random.choice(self.aug_list)
            x = op(x, 3)
            return x

        ## This is the APRS implementation. It always starts with the augmentation of the image and then recombination
        
        op = np.random.choice(self.aug_list)
        x = op(x, 3)

        x_aug = x.copy()
        op = np.random.choice(self.aug_list)
        x_aug = op(x_aug, 3)

        x = np.array(x).astype(np.uint8) 
        x_aug = np.array(x_aug).astype(np.uint8)
        
        fft_1 = np.fft.fftshift(np.fft.fftn(x))
        fft_2 = np.fft.fftshift(np.fft.fftn(x_aug))
        
        abs_1, angle_1 = np.abs(fft_1), np.angle(fft_1)
        abs_2, angle_2 = np.abs(fft_2), np.angle(fft_2)

        fft_1 = abs_1*np.exp((1j) * angle_2)
        fft_2 = abs_2*np.exp((1j) * angle_1)

        p = random.uniform(0, 1)

        if p > 0.5:
            x = np.fft.ifftn(np.fft.ifftshift(fft_1))
        else:
            x = np.fft.ifftn(np.fft.ifftshift(fft_2))

        x = x.astype(np.uint8)
        x = Image.fromarray(x)
        
        return x
