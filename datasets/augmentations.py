# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Base augmentations operators."""

import numpy as np
from PIL import Image, ImageOps, ImageEnhance
import torch
import kornia
import torchvision.transforms as transforms

# ImageNet code should change this value
IMAGE_SIZE = 224#32


def int_parameter(level, maxval):
  """Helper function to scale `val` between 0 and maxval .
  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.
  Returns:
    An int that results from scaling `maxval` according to `level`.
  """
  return int(level * maxval / 10)


def float_parameter(level, maxval):
  """Helper function to scale `val` between 0 and maxval.
  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.
  Returns:
    A float that results from scaling `maxval` according to `level`.
  """
  return float(level) * maxval / 10.


def sample_level(n):
  return np.random.uniform(low=0.1, high=n)

def PILToTorch(x, PILToTensor=False):
   if PILToTensor:
      x = transforms.PILToTensor()(x).unsqueeze(0)
   else:
      x = transforms.ToTensor()(x).unsqueeze(0)
   return x

def TorchToPIL(x):
   x = transforms.ToPILImage()(x.squeeze(0))
   return x



def autocontrast_pil(pil_img, _):
    return ImageOps.autocontrast(pil_img)

def autocontrast(img: torch.Tensor,
                  _=None) -> torch.Tensor:
    
    img = PILToTorch(img)
    with torch.no_grad():
        # BxCxHxW -> BCxHW
        # temporal fix
        reshaped = img.flatten(0, 1).flatten(1, 2).clamp(0, 1) * 255
        # BCx1
        min, _ = reshaped.min(dim=1, keepdim=True)
        max, _ = reshaped.max(dim=1, keepdim=True)
        output = (torch.arange(256, device=img.device, dtype=img.dtype) - min) * (255 / (max - min + 0.1))
        output = output.floor().gather(1, reshaped.long()).reshape_as(img) / 255
    
    output = TorchToPIL(output)
    return output


def equalize_pil(pil_img, _):
    return ImageOps.equalize(pil_img)

def equalize(img: torch.Tensor,
             _=None) -> torch.Tensor:
    
    img = PILToTorch(img)
    # see https://github.com/python-pillow/Pillow/blob/master/src/PIL/ImageOps.py#L319
    with torch.no_grad():
        # BCxHxW
        reshaped = img.clone().flatten(0, 1).clamp_(0, 1) * 255
        size = reshaped.size(0)  # BC
        # 0th channel [0-255], 1st channel [256-511], 2nd channel [512-767]...(BC-1)th channel
        shifted = reshaped + 256 * torch.arange(0, size, device=reshaped.device,
                                                dtype=reshaped.dtype).view(-1, 1, 1)
        # channel wise histogram: BCx256
        histogram = shifted.histc(size * 256, 0, size * 256 - 1).view(size, 256)
        # channel wise cdf: BCx256
        cdf = histogram.cumsum(-1)
        # BCx1
        step = ((cdf[:, -1] - histogram[:, -1]) / 255).floor_().view(size, 1)
        # cdf interpolation, BCx256
        cdf = torch.cat([cdf.new_zeros((cdf.size(0), 1)), cdf], dim=1)[:, :256] + (step / 2).floor_()
        # to avoid zero-div, add 0.1
        output = (cdf / (step + 0.1)).floor_().view(-1)[shifted.long()].reshape_as(img) / 255

    output = TorchToPIL(output)
    return output


def posterize_pil(pil_img, level):
    level = int_parameter(sample_level(level), 4)
    return ImageOps.posterize(pil_img, 4 - level)

def posterize_org(img: torch.Tensor,
              mag: torch.Tensor) -> torch.Tensor:
    
    img = PILToTorch(img)
    mag = torch.Tensor([np.random.uniform(low=0.1, high=1)])
    # mag: 0 to 1
    mag = mag.view(-1, 1, 1, 1)
    with torch.no_grad():
        shift = ((1 - mag) * 8).long()
        shifted = torch.bitwise_right_shift(torch.bitwise_left_shift(img.mul(255).long(), shift), shift)
        #shifted = (img.mul(255).long() << shift) >> shift
    output = shifted.float() / 255, mag

    output = TorchToPIL(output[0])
    return output

def posterize(img: torch.Tensor,
              mag: torch.Tensor) -> torch.Tensor:
    
    img = PILToTorch(img)
    mag = 4 - torch.Tensor([np.random.choice([0,1])])
    # mag: 0 to 1
    output = kornia.enhance.posterize(img, mag)

    output = TorchToPIL(output)
    return output


def rotate_pil(pil_img, level):
  degrees = int_parameter(sample_level(level), 30)
  if np.random.uniform() > 0.5:
    degrees = -degrees
  return pil_img.rotate(degrees, resample=Image.BILINEAR)


def rotate(img: torch.Tensor,
           mag: torch.Tensor) -> torch.Tensor:
    
    mag = torch.Tensor([np.random.uniform(low=1, high=30)])
    #degrees = int_parameter(sample_level(mag), 30)
    if np.random.uniform() > 0.5:
      mag = -1 *mag
    
    img = PILToTorch(img)
    output = kornia.geometry.transform.rotate(img, mag)
    output = TorchToPIL(output)

    return output


def solarize_pil(pil_img, level):
    level = int_parameter(sample_level(level), 256)
    return ImageOps.solarize(pil_img, 256 - level)


def solarize(img: torch.Tensor,
             mag: torch.Tensor) -> torch.Tensor:
    
    img = PILToTorch(img)
    mag = torch.Tensor([np.random.uniform(low=0.1, high=1)])
    mag = mag.view(-1, 1, 1, 1)
    output = torch.where(img < mag, img, 1 - img), mag
    output = TorchToPIL(output[0])
    return output


def shear_x_pil(pil_img, level):
  level = float_parameter(sample_level(level), 0.3)
  if np.random.uniform() > 0.5:
    level = -level
  return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                           Image.AFFINE, (1, level, 0, 0, 1, 0),
                           resample=Image.BILINEAR)


def shear_x(img: torch.Tensor,
            mag: torch.Tensor) -> torch.Tensor:
    
    mag = torch.Tensor([np.random.uniform(low=0.1, high=1)])
    #mag = float_parameter(sample_level(mag), 0.3)
    if np.random.uniform() > 0.5:
      mag = -1 * mag

    img = PILToTorch(img)
    mag = torch.stack([mag, torch.zeros_like(mag)], dim=1)
    output = kornia.geometry.transform.shear(img, mag)
    output = TorchToPIL(output)

    return output


def shear_y_pil(pil_img, level):
  level = float_parameter(sample_level(level), 0.3)
  if np.random.uniform() > 0.5:
    level = -level
  return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                           Image.AFFINE, (1, 0, 0, level, 1, 0),
                           resample=Image.BILINEAR)


def shear_y(img: torch.Tensor,
            mag: torch.Tensor) -> torch.Tensor:
    
    mag = torch.Tensor([np.random.uniform(low=0.1, high=1)])
    #mag = float_parameter(sample_level(mag), 0.3)
    if np.random.uniform() > 0.5:
      mag = -1 * mag

    img = PILToTorch(img)  
    mag = torch.stack([torch.zeros_like(mag), mag], dim=1)
    output = kornia.geometry.transform.shear(img, mag)
    output = TorchToPIL(output)

    return output


def translate_x_pil(pil_img, level):
  level = int_parameter(sample_level(level), IMAGE_SIZE / 3)
  if np.random.random() > 0.5:
    level = -level
  return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                           Image.AFFINE, (1, 0, level, 0, 1, 0),
                           resample=Image.BILINEAR)


def translate_x(img: torch.Tensor,
                mag: torch.Tensor) -> torch.Tensor:
    
    mag = torch.Tensor([np.random.uniform(low=0.05, high=.5)])
    if np.random.random() > 0.5:
      mag = -1* mag
    
    img = PILToTorch(img)  
    mag = torch.stack([mag * img.size(-1), torch.zeros_like(mag)], dim=1)
    output = kornia.geometry.transform.translate(img, mag)
    output = TorchToPIL(output)

    return output

def translate_y_pil(pil_img, level):
  level = int_parameter(sample_level(level), IMAGE_SIZE / 3)
  if np.random.random() > 0.5:
    level = -level
  return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                           Image.AFFINE, (1, 0, 0, 0, 1, level),
                           resample=Image.BILINEAR)


def translate_y(img: torch.Tensor,
                mag: torch.Tensor) -> torch.Tensor:
    

    mag = torch.Tensor([np.random.uniform(low=0.05, high=.5)])
    if np.random.random() > 0.5:
      mag = -1* mag

    img = PILToTorch(img)  
    mag = torch.stack([torch.zeros_like(mag), mag * img.size(-2)], dim=1)
    output = kornia.geometry.translate(img, mag)
    output = TorchToPIL(output)

    return output


# operation that overlaps with ImageNet-C's test set
def color(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Color(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def contrast(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Contrast(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def brightness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Brightness(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def sharpness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Sharpness(pil_img).enhance(level)



## USUALLY MODIFIED: Kornia is a differentiable augmentation (gradient information can be collected)
augmentations_kornia_ver_geo_photo = [
    autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,
    translate_x, translate_y
]

augmentations_kornia_ver_geo = [
    shear_x, shear_y, translate_x, translate_y, rotate
]

augmentations_kornia_ver_photo = [
    autocontrast, equalize, posterize, solarize
]


## USUALLY MODIFIED: Normal image manipulation without gradients
augmentations_pil_ver_geo_photo = [
    autocontrast_pil, equalize_pil, posterize_pil, rotate_pil, solarize_pil, shear_x_pil, shear_y_pil,
    translate_x_pil, translate_y_pil
]

augmentations_pil_ver_geo = [
    rotate_pil, shear_x_pil, shear_y_pil, translate_x_pil, translate_y_pil
]

augmentations_pil_ver_photo = [
    autocontrast_pil, equalize_pil, posterize_pil, solarize_pil
]


## Default Value

augmentations = [
    autocontrast_pil, equalize_pil, posterize_pil, rotate_pil, solarize_pil, shear_x_pil, shear_y_pil,
    translate_x_pil, translate_y_pil
]

#augmentations = [
#    autocontrast, equalize, posterize, solarize
#]

'''augmentations = [
    posterize, solarize
]'''

autocontrast_apply = autocontrast

augmentations_all = [
    autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,
    translate_x, translate_y, color, contrast, brightness, sharpness
]