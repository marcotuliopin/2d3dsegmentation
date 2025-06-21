import torch
import random
import numpy as np
from copy import copy
from torch.utils.data import DataLoader
from utils.dataset import NYUv2
import torchvision.transforms as T
import torchvision.transforms.functional as TF


data_root = "data/nyuv2"

imagenet_mean = (0.485, 0.456, 0.406)
imagenet_std = (0.229, 0.224, 0.225)

nyuv2_rgb_mean = (0.4850, 0.4163, 0.3982)
nyuv2_rgb_std = (0.2878, 0.2952, 0.3087)

nyuv2_hha_mean = (0.538464, 0.4442, 0.4390)
nyuv2_hha_std = (0.2284, 0.2628, 0.1479)

nyuv2_depth_mean = 2.6859
nyuv2_depth_std = 1.2209

height0, width0 = 480, 640  # The original NYUv2 image size


def get_dataloader(
    train: bool = True,
    split_val: bool = False,
    download: bool = False,
    rgb_only: bool = False,
    use_hha: bool = False,
    batch_size: int = 8,
    num_workers: int = 4,
    seed: int = 42,
    image_size: tuple = (height0, width0),
):
    generator = torch.Generator().manual_seed(seed)

    if train:
        dataset = NYUv2(
            data_root,
            seed=seed,
            train=True,
            download=download,
            rgb_transform=train_rgb_transform(),
            seg_transform=train_seg_transform(),
            depth_transform=train_depth_transform() if not rgb_only and not use_hha else None,
            hha_transform=train_hha_transform() if not rgb_only and use_hha else None,
            sync_transform=SyncTransform(*image_size),
        )
    else:
        dataset = NYUv2(
            data_root,
            seed=seed,
            train=False,
            download=download,
            rgb_transform=test_rgb_transform(*image_size),
            seg_transform=test_seg_transform(*image_size),
            depth_transform=test_depth_transform(*image_size) if not rgb_only and not use_hha else None,
            hha_transform=test_hha_transform(*image_size) if not rgb_only and use_hha else None,
            sync_transform=None,
        )

    if split_val and train:
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=generator)
        val_dataset.dataset = copy(dataset)

        val_dataset.dataset.rgb_transform = test_rgb_transform()
        val_dataset.dataset.seg_transform = test_seg_transform()
        val_dataset.dataset.depth_transform = test_depth_transform() if not rgb_only and not use_hha else None
        val_dataset.dataset.hha_transform = test_hha_transform() if not rgb_only and use_hha else None
        val_dataset.dataset.sync_transform = None

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            generator=generator,
            worker_init_fn=worker_init_fn,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            generator=generator,
            worker_init_fn=worker_init_fn,
        )
        return train_loader, val_loader

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        generator=generator,
        worker_init_fn=worker_init_fn,
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
            T.Normalize(mean=[nyuv2_depth_mean], std=[nyuv2_depth_std])
        ]
    )


def train_hha_transform():
    return T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=nyuv2_hha_mean, std=nyuv2_hha_std),
        ]
    )


def test_rgb_transform(height=height0, width=width0):
    return T.Compose(
        [
            T.CenterCrop((height, width)),
            T.ToTensor(),
            T.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )


def test_seg_transform(height=height0, width=width0):
    return T.Compose(
        [
            T.CenterCrop((height, width)),
            T.ToTensor(),
            TensorToLongMask(),  # Convert to long type for segmentation masks
        ]
    )


def test_depth_transform(height=height0, width=width0):
    return T.Compose(
        [
            T.CenterCrop((height, width)),
            DepthToTensor(),
            T.Normalize(mean=[nyuv2_depth_mean], std=[nyuv2_depth_std])
        ]
    )


def test_hha_transform(height=height0, width=width0):
    return T.Compose(
        [
            T.CenterCrop((height, width)),
            T.ToTensor(),
            T.Normalize(mean=nyuv2_hha_mean, std=nyuv2_hha_std),
        ]
    )


class SyncTransform:
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, rgb_img, seg_mask, depth_img=None, hha_img=None):
        i, j, h, w = T.RandomCrop.get_params(rgb_img, (self.height, self.width))
        rgb_img = TF.crop(rgb_img, i, j, h, w)
        seg_mask = TF.crop(seg_mask, i, j, h, w)
        if depth_img is not None:
            depth_img = TF.crop(depth_img, i, j, h, w)
        if hha_img is not None:
            hha_img = TF.crop(hha_img, i, j, h, w)

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

    def _depth_to_tensor(self, depth_img):
        depth_np = np.array(depth_img, dtype=np.uint16)
        depth_meters = depth_np.astype(np.float32) / 1e4 # Convert to meters
        depth_tensor = torch.from_numpy(depth_meters).unsqueeze(0)  # (1, H, W)
        return depth_tensor
