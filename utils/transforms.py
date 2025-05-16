import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_training_transforms(height=256, width=352):
    return A.Compose([
        A.Resize(height=height, width=width, interpolation=cv2.INTER_LINEAR),
        A.HorizontalFlip(p=0.3),
        A.GridDistortion(p=0.4),
        A.RandomBrightnessContrast(p=0.2),
        A.Affine(
            scale=(0.8, 1.2),
            translate_percent=(-0.1, 0.1),
            rotate=(-10, 10),
            p=0.5
        ),
        A.GaussNoise(p=0.4),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], additional_targets={'mask': 'mask'})

def get_validation_transforms(height=256, width=352):
    return A.Compose([
        A.Resize(height=height, width=width, interpolation=cv2.INTER_NEAREST),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], additional_targets={'mask': 'mask'})