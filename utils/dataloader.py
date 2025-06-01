import torch
import random
import numpy as np
from copy import copy
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.dataset import NYUv2

data_root = "data/nyuv2"

nyuv2_depth_mean = 2.684067
nyuv2_depth_std = 0.921854

# The mean and std values for ImageNet, used for normalization
# with pretrained models.
imagenet_mean = (0.485, 0.456, 0.406)
imagenet_std = (0.229, 0.224, 0.225)

height0, width0 = 480, 640  # The original NYUv2 image size


def nyuv2_dataloader(
    train: bool = True,
    split_val: bool = False,
    download: bool = False,
    rgb_only: bool = False,
    hha: bool = False,
    batch_size: int = 16,
    num_workers: int = 0,
    image_size: tuple = (height0, width0),
    seed: int = 42,
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

    rgb_xform = train_rgb_transform(*image_size) if train else test_rgb_transform(*image_size)
    seg_xform = train_seg_transform(*image_size) if train else test_seg_transform(*image_size)
    depth_xform = train_depth_transform(*image_size) if train else test_depth_transform(*image_size)
    depth_xform = None if rgb_only else depth_xform
    hha_xform = None if rgb_only else train_hha_transform(*image_size) if hha else None

    dataset = NYUv2(
        data_root,
        seed=seed,
        train=train,
        download=download,
        rgb_transform=rgb_xform,
        seg_transform=seg_xform,
        depth_transform=depth_xform,
        hha_transform=hha_xform,
    )

    if split_val:
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=generator)
        val_dataset.dataset = copy(dataset)

        # Remove augmentation from validation set
        val_dataset.dataset.rgb_transform = test_rgb_transform(*image_size)

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
    )


def worker_init_fn(worker_id):
    """
    Initializes the worker process with a unique seed based on the worker ID.
    """
    seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def train_rgb_transform(height=height0, width=width0):
    """
    Transformation for train RGB images. Apply augmentation techniques.
    """
    return transforms.Compose(
        [
            transforms.Resize((height, width), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )


def train_seg_transform(height=height0, width=width0):
    """
    Transformation for segmentation masks.
    """
    return transforms.Compose(
        [
            transforms.Resize((height, width), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ]
    )


def train_depth_transform(height=height0, width=width0):
    """
    Transformation for depth images.
    """
    return transforms.Compose(
        [
            transforms.Resize((height, width), interpolation=transforms.InterpolationMode.NEAREST),
            DepthToTensor(),
            transforms.Normalize(mean=[nyuv2_depth_mean], std=[nyuv2_depth_std])
        ]
    )


def test_rgb_transform(height=height0, width=width0):
    """
    Transformation for validation/test RGB images.
    """
    return transforms.Compose(
        [
            transforms.Resize((height, width), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )


def test_seg_transform(height=height0, width=width0):
    """
    Transformation for validation/test segmentation masks.
    """
    return transforms.Compose(
        [
            transforms.Resize((height, width), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ]
    )


def test_depth_transform(height=height0, width=width0):
    """
    Transformation for validation/test depth images.
    """
    return transforms.Compose(
        [
            transforms.Resize((height, width), interpolation=transforms.InterpolationMode.NEAREST),
            DepthToTensor(),
            transforms.Normalize(mean=[nyuv2_depth_mean], std=[nyuv2_depth_std])
        ]
    )


def train_hha_transform(height=height0, width=width0):
    """
    Transformation for HHA images (depth-based features).
    """
    pass


class DepthToTensor:
    """
    Custom transform to convert depth images to tensors.
    """
    def __call__(self, depth_img):
        return depth_to_tensor(depth_img)


def depth_to_tensor(depth_img):
    """
    Convert depth image to tensor.
    :param sample: PIL image uint16
    """
    depth_np = np.array(depth_img, dtype=np.uint16)
    depth_meters = depth_np.astype(np.float32) / 1e4
    depth_tensor = torch.from_numpy(depth_meters).unsqueeze(0)  # (1, H, W)
    return depth_tensor
    