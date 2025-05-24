import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

# TODO: Run without data augmentation
def get_training_transforms(height, width):
    return A.Compose([
        A.RandomResizedCrop(size=(height, width), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        A.HorizontalFlip(p=0.5),
        A.GridDistortion(p=0.4),
        A.RandomBrightnessContrast(p=0.4),
        A.ColorJitter(p=0.4),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], additional_targets={'mask': 'mask'})

def get_validation_transforms(height, width):
    return A.Compose([
        A.Resize(height=height, width=width, interpolation=cv2.INTER_NEAREST),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], additional_targets={'mask': 'mask'})