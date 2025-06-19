import torch
import random
import numpy as np
from copy import copy
from torch.utils.data import DataLoader
from torchvision import transforms
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
    """
    Creates a DataLoader for the NYUv2 dataset.

    :param train: whether to load the train or test set
    :param split_val: whether to split the training set into train and validation sets
    :param download: whether to download and process data if missing
    :param rgb_only: whether to include depth images in the dataset
    :param batch_size: number of samples per batch
    :param num_workers: number of subprocesses to use for data loading
    :param image_size: size of the images (height, width)
    :param seed: random seed for reproducibility
    """
    generator = torch.Generator().manual_seed(seed)

    rgb_xform = train_rgb_transform() if train else test_rgb_transform(*image_size)
    seg_xform = train_seg_transform() if train else test_seg_transform(*image_size)
    depth_xform = train_depth_transform() if train else test_depth_transform(*image_size)
    depth_xform = None
    hha_xform = None
    sync_transform = SyncTransform(*image_size) if train else None

    if not rgb_only:
        if use_hha:
            hha_xform = train_hha_transform() if train else test_hha_transform(*image_size)
        else:
            depth_xform = train_depth_transform() if train else test_depth_transform(*image_size)

    dataset = NYUv2(
        data_root,
        seed=seed,
        train=train,
        download=download,
        rgb_transform=rgb_xform,
        seg_transform=seg_xform,
        depth_transform=depth_xform,
        hha_transform=hha_xform,
        sync_transform=sync_transform,
    )

    if split_val and train:
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=generator)
        val_dataset.dataset = copy(dataset)

        val_dataset.dataset.rgb_transform = test_rgb_transform()
        val_dataset.dataset.seg_transform = test_seg_transform()
        if depth_xform is not None:
            val_dataset.dataset.depth_transform = test_depth_transform()
        if hha_xform is not None:
            val_dataset.dataset.hha_transform = test_hha_transform()
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
    return transforms.Compose(
        [
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=nyuv2_rgb_mean, std=nyuv2_rgb_std),
        ]
    )


def train_seg_transform():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            TensorToLongMask(),  # Convert to long type for segmentation masks
        ]
    )


def train_depth_transform():
    return transforms.Compose(
        [
            DepthToTensor(),
            transforms.Normalize(mean=[nyuv2_depth_mean], std=[nyuv2_depth_std])
        ]
    )


def train_hha_transform():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=nyuv2_hha_mean, std=nyuv2_hha_std),
        ]
    )


def test_rgb_transform(height=height0, width=width0):
    return transforms.Compose(
        [
            T.CenterCrop((height, width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=nyuv2_rgb_mean, std=nyuv2_rgb_std),
        ]
    )


def test_seg_transform(height=height0, width=width0):
    return transforms.Compose(
        [
            T.CenterCrop((height, width)),
            transforms.ToTensor(),
            TensorToLongMask(),  # Convert to long type for segmentation masks
        ]
    )


def test_depth_transform(height=height0, width=width0):
    return transforms.Compose(
        [
            T.CenterCrop((height, width)),
            DepthToTensor(),
            transforms.Normalize(mean=[nyuv2_depth_mean], std=[nyuv2_depth_std])
        ]
    )


def test_hha_transform(height=height0, width=width0):
    return transforms.Compose(
        [
            T.CenterCrop((height, width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=nyuv2_hha_mean, std=nyuv2_hha_std),
        ]
    )


class SyncTransform:
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, rgb_img, seg_mask, depth_img=None, hha_img=None):
        i, j, h, w = T.RandomResizedCrop.get_params(rgb_img, scale=(0.8, 1.0), ratio=(0.9, 1.2))
        rgb_img = TF.resized_crop(rgb_img, i, j, h, w, (self.height, self.width), T.InterpolationMode.BILINEAR)
        seg_mask = TF.resized_crop(seg_mask, i, j, h, w, (self.height, self.width), T.InterpolationMode.NEAREST)
        if depth_img is not None:
            depth_img = TF.resized_crop(depth_img, i, j, h, w, (self.height, self.width), T.InterpolationMode.NEAREST)
        if hha_img is not None:
            hha_img = TF.resized_crop(hha_img, i, j, h, w, (self.height, self.width), T.InterpolationMode.NEAREST)

        # 2. Aplicar RandomHorizontalFlip
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
    """
    Custom transform to convert tensor masks to long type.
    """
    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            # ToTensor scales to [0, 1] by default
            img = (img * 255).long()
            img = img.squeeze()
        return img


class DepthToTensor:
    """
    Custom transform to convert depth images to tensors.
    """
    def __call__(self, depth_img):
        return depth_to_tensor(depth_img)


def depth_to_tensor(depth_img):
    """
    Convert depth image to tensor.
    :param depth_img: PIL image uint16 with depth in mm
    """
    depth_np = np.array(depth_img, dtype=np.uint16)
    depth_meters = depth_np.astype(np.float32) / 1e4 # Convert to meters
    depth_tensor = torch.from_numpy(depth_meters).unsqueeze(0)  # (1, H, W)
    return depth_tensor
