import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

nyuv2_depth_mean = 156.45574834168633
nyuv2_depth_std = 86.42665851132719

imagenet_mean = (0.485, 0.456, 0.406)
imagenet_std = (0.229, 0.224, 0.225)

def get_training_transforms(height, width):
    return A.Compose([
        A.Resize(height=height, width=width, interpolation=cv2.INTER_LINEAR),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=10, p=0.5),
        # A.ColorJitter(p=0.4),
        # A.RandomBrightnessContrast(p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406, nyuv2_depth_mean), std=(0.229, 0.224, 0.225, nyuv2_depth_std)),
        ToTensorV2(),
    ], additional_targets={'mask': 'mask'})

def get_validation_transforms(height, width):
    return A.Compose([
        A.Resize(height=height, width=width, interpolation=cv2.INTER_LINEAR),
        A.Normalize(mean=(0.485, 0.456, 0.406, nyuv2_depth_mean), std=(0.229, 0.224, 0.225, nyuv2_depth_std)),
        ToTensorV2(),
    ], additional_targets={'mask': 'mask'})