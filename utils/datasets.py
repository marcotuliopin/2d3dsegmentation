import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class SunRGBDDataset(Dataset):
    def __init__(self, path_file, root_dir="", transform=None):
        self.transform = transform
        self.path_df = pd.read_csv(path_file, sep=' ', header=None)
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
            image = transformed['image']
            mask = transformed['mask']
            mask = mask.long()

        return image, mask