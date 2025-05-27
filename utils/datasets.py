import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from datasets import load_dataset

from utils.transforms import resize_transform, train_rgb_transform, val_rgb_transform, depth_transform, shared_augmentation_transform


class SunRGBDDataset(Dataset):
    def __init__(self, path_file, root_dir="", transform=None):
        self.transform = transform
        self.path_df = pd.read_csv(path_file, sep=" ", header=None)
        self.root_dir = root_dir

    def __len__(self):
        return len(self.path_df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.path_df.iloc[idx, 0])
        mask_path = os.path.join(self.root_dir, self.path_df.iloc[idx, 2])
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            transformed = self.transform(image=np.array(image), mask=np.array(mask))
            image = transformed["image"]
            mask = transformed["mask"]
            mask = mask.long()

        return image, mask


class NYUDepthV2Dataset(Dataset):
    def __init__(self, file_path, shape, mode="train", ignore_index=255, use_depth=False):
        self.data = load_dataset("parquet", data_files=file_path)["train"]
        self.shape = shape
        self.mode = mode
        self.ignore_index = ignore_index
        self.use_depth = use_depth
        self.num_classes = 40

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        rgb = np.array(sample["image"].convert("RGB"))
        mask = np.array(sample["label"].convert("L"))
        depth = np.array(sample["depth"].convert("L"))[..., np.newaxis]  # (H, W) → (H, W, 1)

        # Mask preprocessing:
        # 1. Mapping values:
        #    - 0 (unlabeled) → -1 (ignore_index)
        #    - 1-40 (valid classes) → 0-39
        mask[mask == 0] = self.ignore_index  # Unlabeled
        valid_classes_mask = (mask >= 1) & (mask != self.ignore_index)
        mask[valid_classes_mask] -= 1  # Valid classes: 1-40 → 0-39
    
        resized = resize_transform(self.shape[0], self.shape[1])(image=rgb, depth=depth, mask=mask)
        rgb = resized["image"]
        mask = resized["mask"]
        depth = resized["depth"]

        if self.mode == "train":
            augmented = shared_augmentation_transform(image=rgb, depth=depth, mask=mask)
            transformed = train_rgb_transform(image=augmented["image"], mask=augmented["mask"])
        else:
            transformed = val_rgb_transform(image=rgb, mask=mask)

        rgb = transformed["image"]
        mask = transformed["mask"]
        depth = depth_transform(image=augmented["depth"])["image"]

        if not self.use_depth:
            return rgb, mask.long()

        image = torch.cat((rgb, depth), dim=0)
        return image, mask.long()


def calculate_class_weights(dataloader, num_classes, ignore_index):
    class_counts = np.zeros(num_classes)
    for _, targets in dataloader:
        valid_pixels = targets[targets != ignore_index]
        unique, counts = torch.unique(valid_pixels, return_counts=True)
        for cls, cnt in zip(unique, counts):
            class_counts[cls] += cnt.item()
    weights = 1.0 / (class_counts + 1e-6)
    weights = weights / weights.sum()
    return torch.tensor(weights, dtype=torch.float32)