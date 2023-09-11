from PIL import Image
import numpy as np
import datasets.augmentations as augmentations
from common_corruption.imagecorruptions import corrupt

class ComCorAugmentationsApply(object):
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
        
        ## normal augmentations

        # p = random.uniform(0, 1)
        # if p > (1-self.add_aug):
        #     op = np.random.choice(self.aug_list)
        #     x = op(x, 3)

        ## common corruptions augmentations

        corruption_choice = np.random.choice(15)
        severity = np.random.choice(5, p=[.15,.22,.26,.22,.15]) + 1
        x = np.array(x).astype(np.uint8) 
        corrupted_image = corrupt(x, corruption_number=corruption_choice, severity=severity)
        x = corrupted_image
        x = Image.fromarray(x)

    
        
        return x
