from copy import copy
import os
import random
from PIL import Image
import numpy as np

import torchvision.transforms as T
import torchvision.transforms.functional as TF

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


imagenet_mean = (0.485, 0.456, 0.406)
imagenet_std = (0.229, 0.224, 0.225)


class SunRGBD(Dataset):
    def __init__(
        self,
        root: str = "data/sunrgbd",
        seed: int = 42,
        train: bool = True,
        rgb_transform: bool = None,
        seg_transform: bool = None,
        depth_transform: bool = None,
        hha_transform: bool = None,
        sync_transform: bool = None, 
    ):
        super().__init__()
        self.root = root
        self.seed = seed
        self.train = train

        self.rgb_transform = rgb_transform
        self.seg_transform = seg_transform
        self.depth_transform = depth_transform
        self.hha_transform = hha_transform
        self.sync_transform = sync_transform

        self._split = "train" if train else "test"

        self._rgb_files = sorted(os.listdir(os.path.join(root, "image", self._split)))
        self._depth_files = sorted(os.listdir(os.path.join(root, "depth", self._split)))
        self._seg_files = sorted(os.listdir(os.path.join(root, "label37", self._split)))
        self._hha_files = sorted(os.listdir(os.path.join(root, "hha", self._split)))
    
    def __getitem__(self, index: int):
        folder = lambda x: os.path.join(self.root, x, self._split)
        imgs = []

        if self.rgb_transform is not None:
            img = Image.open(os.path.join(folder("image"), self._rgb_files[index]))
            imgs.append(img)

        if self.depth_transform is not None:
            img = Image.open(os.path.join(folder("depth"), self._depth_files[index]))
            imgs.append(img)

        if self.seg_transform is not None:
            img = Image.open(os.path.join(folder("label37"), self._seg_files[index]))
            imgs.append(img)

        if self.hha_transform is not None:
            img = Image.open(os.path.join(folder("hha"), self._hha_files[index]))
            imgs.append(img)

        return self._transform(imgs)
    
    def __len__(self):
        return len(self._rgb_files)
    
    def _transform(self, imgs):
        if self.sync_transform is not None:
            return self.sync_transform(
                imgs[0],
                imgs[1],
                imgs[2] if self.depth_transform else None,
                imgs[2] if self.hha_transform else None,
            )
        
        if self.rgb_transform is not None:
            imgs[0] = self.rgb_transform(imgs[0])
        if self.seg_transform is not None:
            imgs[1] = self.seg_transform(imgs[1])
        if self.depth_transform is not None:
            imgs[2] = self.depth_transform(imgs[2])
        if self.hha_transform is not None:
            imgs[2] = self.hha_transform(imgs[2])

        return imgs


def get_dataloader(
    root: str = "data/sunrgbd",
    seed: int = 42,
    img_height: int = 256,
    img_width: int = 256,
    batch_size: int = 32,
    num_workers: int = 4,
    train: bool = True,
    use_val: bool = True,
    rgb_only: bool = False,
    use_hha: bool = False,
):
    generator = torch.Generator().manual_seed(seed)

    if train:
        dataset = SunRGBD(
            root=root,
            seed=seed,
            train=True,
            rgb_transform=train_rgb_transform(),
            seg_transform=train_seg_transform(),
            depth_transform=train_depth_transform() if not rgb_only and not use_hha else None,
            hha_transform=train_hha_transform() if not rgb_only and use_hha else None,
        )
    else:
        dataset = SunRGBD(
            root=root,
            seed=seed,
            train=False,
            rgb_transform=test_rgb_transform(),
            seg_transform=test_seg_transform(),
            depth_transform=test_depth_transform() if not rgb_only and not use_hha else None,
            hha_transform=test_hha_transform() if not rgb_only and use_hha else None,
        )
    
    if train and use_val:
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=generator)

        val_dataset.dataset = copy(dataset)
        val_dataset.dataset.rgb_transform = test_rgb_transform(img_height, img_width)
        val_dataset.dataset.seg_transform = test_seg_transform(img_height, img_width)
        val_dataset.dataset.depth_transform = test_depth_transform(img_height, img_width) if not rgb_only and not use_hha else None
        val_dataset.dataset.hha_transform = test_hha_transform(img_height, img_width) if not rgb_only and use_hha else None
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            worker_init_fn=worker_init_fn,
            generator=generator,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            worker_init_fn=worker_init_fn,
            generator=generator,
        )
        return train_loader, val_loader
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        generator=generator,
    )
        

def worker_init_fn(worker_id):
    seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def train_rgb_transform():
    return T.Compose(
        [
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.1),
            T.ToTensor(),
            T.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )


def train_seg_transform():
    return T.Compose(
        [
            T.ToTensor(),
            TensorToLongMask(),  # Convert to long type for segmentation masks
        ]
    )


def train_depth_transform():
    return T.Compose(
        [
            DepthToTensor(),
            T.Normalize(mean=[sun_depth_mean], std=[sun_depth_std])
        ]
    )


def train_hha_transform(height, width):
    return T.Compose(
        [
            # transforms.Resize((height, width), interpolation=transforms.InterpolationMode.NEAREST),
            T.ToTensor(),
            T.Normalize(mean=sun_hha_mean, std=sun_hha_std),
        ]
    )


def test_rgb_transform(height, width):
    return T.Compose(
        [
            T.Resize((height, width), interpolation=T.InterpolationMode.BILINEAR),
            # transforms.CenterCrop((height, width)),
            T.ToTensor(),
            T.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )


def test_seg_transform(height, width):
    return T.Compose(
        [
            T.Resize((height, width), interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor(),
            TensorToLongMask(),  # Convert to long type for segmentation masks
        ]
    )


def test_depth_transform(height, width):
    return T.Compose(
        [
            T.Resize((height, width), interpolation=T.InterpolationMode.NEAREST),
            # transforms.CenterCrop((height, width)),
            DepthToTensor(),
            T.Normalize(mean=[sun_depth_mean], std=[sun_depth_std])
        ]
    )


def test_hha_transform(height, width):
    return T.Compose(
        [
            T.Resize((height, width), interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor(),
            T.Normalize(mean=sun_hha_mean, std=sun_hha_std),
        ]
    )


class SyncTransform:
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, rgb_img, seg_mask, depth_img=None, hha_img=None):
        i, j, h, w = T.RandomResizedCrop.get_params(rgb_img, scale=(0.7, 1.0), ratio=(0.75, 1.33))
        rgb_img = TF.resized_crop(rgb_img, i, j, h, w, (self.height, self.width), T.InterpolationMode.BILINEAR)
        seg_mask = TF.resized_crop(seg_mask, i, j, h, w, (self.height, self.width), T.InterpolationMode.NEAREST)
        if depth_img is not None:
            depth_img = TF.resized_crop(depth_img, i, j, h, w, (self.height, self.width), T.InterpolationMode.NEAREST)
        if hha_img is not None:
            hha_img = TF.resized_crop(hha_img, i, j, h, w, (self.height, self.width), T.InterpolationMode.NEAREST)

        if random.random() < 0.5:
            rgb_img = TF.hflip(rgb_img)
            seg_mask = TF.hflip(seg_mask)
            if depth_img is not None:
                depth_img = TF.hflip(depth_img)
            if hha_img is not None:
                hha_img = TF.hflip(hha_img)

        imgs = [rgb_img, seg_mask]
        if depth_img is not None:
            imgs.append(depth_img)
        if hha_img is not None:
            imgs.append(hha_img)

        return imgs


class TensorToLongMask(object):
    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            # ToTensor scales to [0, 1] by default
            img = (img * 255).long()
            img = img.squeeze()
        return img


class DepthToTensor:
    def __call__(self, depth_img):
        return self._depth_to_tensor(depth_img)

    def _depth_to_tensor(depth_img):
        depth_np = np.array(depth_img, dtype=np.uint16)
        depth_meters = depth_np.astype(np.float32) / 1e4 # Convert to meters
        depth_tensor = torch.from_numpy(depth_meters).unsqueeze(0)  # (1, H, W)
        return depth_tensor
