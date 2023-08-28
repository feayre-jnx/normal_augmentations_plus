import io
import random
from PIL import Image
import numpy as np
import datasets.augmentations as augmentations
import albumentations as A

class AlbumentationsApply(object):
    def __init__(self, img_size=32, aug=None, alb_policy=None):
        self.img_size = img_size
        if aug is None:
            augmentations.IMAGE_SIZE = img_size     ## Does not have any meaning, change the value directly from the function
            self.aug_list = augmentations.augmentations
        else:
            self.aug_list = aug.augmentations

        self.alb_policy = alb_policy

        ## prepare unnormalize code
        self.remove_norm = UnNormalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))#(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    def __call__(self, x):
        '''
        :param img: (PIL Image): Image
        :return: code img (PIL Image): Image
        '''
        
        p = .1#random.uniform(0, 1)
        if p > 0.5:
            return x
        
        if self.alb_policy == None:
            
            ## declare the Albumentation Augmentations
            albumentations_sel = ['autocontrast', 'equalize', 'rotate', 'posterize',
                            'solarize', 'shear_x', 'shear_y', 'translate_x', 'translate_y']
            selected_aug = np.random.choice(albumentations_sel)

            level_rate_high = 3
            level = np.random.uniform(low=0.1, high=level_rate_high)

            if selected_aug == 'autocontrast':
                x = augmentations.autocontrast_apply(x, 3)
            elif selected_aug == 'equalize':
                transform_alb = A.Compose([A.Equalize()])
            elif selected_aug == 'posterize':
                level = int(level * 4 / 10)
                transform_alb = A.Compose([A.Posterize(4-level)])   
            elif selected_aug == 'rotate':
                degrees = int(level * 30 / 10)
                if np.random.uniform() > 0.5:
                    degrees = -degrees
                transform_alb = A.Compose([A.Rotate(degrees,interpolation=2)])
            elif selected_aug == 'solarize':
                level = int(level * 256 / 10)
                transform_alb = A.Compose([A.Solarize(256-level)])
            elif selected_aug == 'shear_x':
                level = float(level) * 0.3 / 10
                if np.random.uniform() > 0.5:
                    level = -level
                transform_alb = A.Compose([A.Affine(shear={'x':level}, interpolation=2)])
            elif selected_aug == 'shear_y':
                level = float(level) * 0.3 / 10
                if np.random.uniform() > 0.5:
                    level = -level
                transform_alb = A.Compose([A.Affine(shear={'y':level}, interpolation=2)])
            elif selected_aug == 'translate_x':
                level = int(level * (self.img_size/3) / 10)
                if np.random.uniform() > 0.5:
                    level = -level
                transform_alb = A.Compose([A.Affine(translate_px={'x':level}, interpolation=2)])
            elif selected_aug == 'translate_y':
                level = int(level * (self.img_size/3) / 10)
                if np.random.uniform() > 0.5:
                    level = -level
                transform_alb = A.Compose([A.Affine(translate_px={'y':level}, interpolation=2)])

            if selected_aug != 'autocontrast':

                ## convert PIL image to cv2 format (RGB)
                x = np.asarray(x)
                ## apply the transform
                x = transform_alb(image=x)['image']
                ## return the image back to PIL
                x = Image.fromarray(x)


            return x
        
        else:
            ## apply albumentation augmentations from the policy
            
            ## convert PIL image to cv2 format (RGB)
            x = np.asarray(x)
            ## apply the transform
            transform_alb = A.load(self.alb_policy)
            x = transform_alb(image=x)['image']
            ## remove normalization from the policy
            x = self.remove_norm(x)

            ## return the image back to PIL
            #x = Image.fromarray(x)
            import torchvision.transforms as T
            x = T.ToPILImage()(x)

            return x
        

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor
        