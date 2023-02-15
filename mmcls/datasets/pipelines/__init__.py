# Copyright (c) OpenMMLab. All rights reserved.
from .auto_augment import (AutoAugment, AutoContrast, Brightness,
                           ColorTransform, Contrast, Cutout, Equalize, Invert,
                           Posterize, RandAugment, Rotate, Sharpness, Shear,
                           Solarize, SolarizeAdd, Translate)
from .compose import Compose
from .formatting import (Collect, ImageToTensor, ToNumpy, ToPIL, ToTensor,
                         Transpose, to_tensor)
from .loading import LoadImageFromFile
from .transforms import (CenterCrop, ColorJitter, Lighting, Normalize, Pad,
                         RandomCrop, RandomErasing, RandomFlip,
                         RandomGrayscale, RandomResizedCrop, Resize)

__all__ = [
    'Compose', 'to_tensor', 'ToTensor', 'ImageToTensor', 'ToPIL', 'ToNumpy',
    'Transpose', 'Collect', 'LoadImageFromFile', 'Resize', 'CenterCrop',
    'RandomFlip', 'Normalize', 'RandomCrop', 'RandomResizedCrop',
    'RandomGrayscale', 'Shear', 'Translate', 'Rotate', 'Invert',
    'ColorTransform', 'Solarize', 'Posterize', 'AutoContrast', 'Equalize',
    'Contrast', 'Brightness', 'Sharpness', 'AutoAugment', 'SolarizeAdd',
    'Cutout', 'RandAugment', 'Lighting', 'ColorJitter', 'RandomErasing', 'Pad'
]
'''
CenterCrop 中心裁减 从中心开始 裁出指定长度宽度的图片
ColorJitter 
Lighting 
Normalize
Pad
RandomCrop 随机裁剪
RandomErasing
RandomFlip
RandomGrayscale
RandomResizedCrop resize 和 crop组合
Resize resize
'''
