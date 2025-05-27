import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

nyuv2_depth_mean = 156.45574834168633
nyuv2_depth_std = 86.42665851132719

imagenet_mean = (0.485, 0.456, 0.406)
imagenet_std = (0.229, 0.224, 0.225)
    
def resize_transform(height, width):
    return A.Compose([
        A.Resize(height=height, width=width, interpolation=cv2.INTER_LINEAR),
    ], additional_targets={
        'depth': 'image',
        'mask': 'mask'
    })

shared_augmentation_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=10, p=0.5),
    ], additional_targets={
        'depth': 'image',
        'mask': 'mask'
    })

train_rgb_transform = A.Compose([
        A.ColorJitter(p=0.4),
        A.RandomBrightnessContrast(p=0.3),
        A.Normalize(mean=imagenet_mean, std=imagenet_std),
        ToTensorV2(),
    ], additional_targets={'mask': 'mask'})

val_rgb_transform = A.Compose([
        A.Normalize(mean=imagenet_mean, std=imagenet_std),
        ToTensorV2(),
    ], additional_targets={'mask': 'mask'})

depth_transform = A.Compose([
        A.Normalize(mean=nyuv2_depth_mean, std=nyuv2_depth_std),
        ToTensorV2()
    ])