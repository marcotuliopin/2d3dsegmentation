import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from datasets import load_dataset


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
    def __init__(self, path_file, transform=None, split_name="train", ignore_index=255, use_depth=False):
        self.data = load_dataset("parquet", data_files={split_name: path_file})
        self.split_name = split_name
        self.transform = transform
        self.num_classes = 40
        self.ignore_index = ignore_index
        self.use_depth = use_depth

    def __len__(self):
        return len(self.data[self.split_name])

    def __getitem__(self, idx):
        sample = self.data[self.split_name][idx]
        image = np.array(sample["image"].convert("RGB"))
        mask = np.array(sample["label"].convert("L"))

        if self.use_depth:
            depth = np.array(sample["depth"].convert("L"))
            depth = depth[..., np.newaxis]  # (H, W) → (H, W, 1)
            image = np.concatenate((image, depth), axis=2)

        # --- MASK PREPROCESSING ---
        # 1. Mapping values:
        #    - 0 (unlabeled) → -1 (ignore_index)
        #    - 1-40 (valid classes) → 0-39
        mask[mask == 0] = self.ignore_index  # Unlabeled
        valid_classes_mask = (mask >= 1) & (mask != self.ignore_index)
        mask[valid_classes_mask] -= 1  # Valid classes: 1-40 → 0-39

        if self.transform:
            transformed = self.transform(image=np.array(image), mask=np.array(mask))
            image = transformed["image"]
            mask = transformed["mask"]
            mask = mask.long()

        return image, mask


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