import os
import h5py
import torch
import shutil
import tarfile
import zipfile
import requests
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url

"""
author: Mihai Suteu
date: 15/05/19
"""


class NYUv2(Dataset):
    """
    PyTorch wrapper for the NYUv2 dataset focused on multi-task learning.
    Data sources available: RGB, Semantic Segmentation, Surface Normals, Depth Images.
    If no transformation is provided, the image type will not be returned.

    ### Output
    All images are of size: 640 x 480

    1. RGB: 3 channel input image

    2. Semantic Segmentation: 1 channel representing one of the 14 (0 -
    background) classes. Conversion to int will happen automatically if
    transformation ends in a tensor.

    3. Surface Normals: 3 channels, with values in [0, 1].

    4. Depth Images: 1 channel with floats representing the distance in meters.
    Conversion will happen automatically if transformation ends in a tensor.
    """

    def __init__(
        self,
        root: str,
        seed: int = 42,
        train: bool = True,
        download: bool = False,
        rgb_transform=None,
        seg_transform=None,
        sn_transform=None,
        depth_transform=None,
        hha_transform=None,
        sync_transform=None,
    ):
        """
        Will return tuples based on what data source has been enabled (rgb, seg etc).

        :param root: path to root folder (eg /data/NYUv2)
        :param train: whether to load the train or test set
        :param download: whether to download and process data if missing
        :param rgb_transform: the transformation pipeline for rbg images
        :param seg_transform: the transformation pipeline for segmentation images. If
        the transformation ends in a tensor, the result will be automatically
        converted to int in [0, 14)
        :param sn_transform: the transformation pipeline for surface normal images
        :param depth_transform: the transformation pipeline for depth images. If the
        transformation ends in a tensor, the result will be automatically converted
        to meters
        :param hha_transform: the transformation pipeline for HHA images
        :param sync_transform: a transformation that will be applied to all images
        """
        super().__init__()
        self.root = root
        self.seed = seed

        self.rgb_transform = rgb_transform
        self.seg_transform = seg_transform
        self.sn_transform = sn_transform
        self.depth_transform = depth_transform
        self.hha_transform = hha_transform
        self.sync_transform = sync_transform

        self.train = train
        self._split = "train" if train else "test"

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not complete." + " You can use download=True to download it")

        # rgb folder as ground truth
        self._files = sorted(os.listdir(os.path.join(root, f"{self._split}_rgb")))

    def __getitem__(self, index: int):
        folder = lambda name: os.path.join(self.root, f"{self._split}_{name}")
        imgs = []

        if self.rgb_transform is not None:
            img = Image.open(os.path.join(folder("rgb"), self._files[index]))
            imgs.append(img)

        if self.seg_transform is not None:
            img = Image.open(os.path.join(folder("seg13"), self._files[index]))
            imgs.append(img)

        if self.depth_transform is not None:
            img = Image.open(os.path.join(folder("depth"), self._files[index]))
            imgs.append(img)
        
        if self.hha_transform is not None:
            file_index = self._files[index].split('.')[0]  
            hha_filename = f"{file_index}_hha.png"
            img = Image.open(os.path.join(folder("hha"), hha_filename))
            imgs.append(img)
        
        imgs = self._augment(imgs) 

        return imgs

    def __len__(self):
        return len(self._files)
    
    def _augment(self, imgs):
        if self.sync_transform is not None:
            imgs = self.sync_transform(
                imgs[0],
                imgs[1],
                imgs[2] if self.depth_transform is not None else None,
                imgs[2] if self.hha_transform is not None else None,
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

    def __repr__(self):
        fmt_str = f"Dataset {self.__class__.__name__}\n"
        fmt_str += f"    Number of data points: {self.__len__()}\n"
        fmt_str += f"    Split: {self._split}\n"
        fmt_str += f"    Root Location: {self.root}\n"
        tmp = "    RGB Transforms: "
        fmt_str += "{0}{1}\n".format(
            tmp, self.rgb_transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )
        tmp = "    Seg Transforms: "
        fmt_str += "{0}{1}\n".format(
            tmp, self.seg_transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )
        tmp = "    SN Transforms: "
        fmt_str += "{0}{1}\n".format(
            tmp, self.sn_transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )
        tmp = "    Depth Transforms: "
        fmt_str += "{0}{1}\n".format(
            tmp, self.depth_transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )
        tmp = "    HHA Transforms: "
        fmt_str += "{0}{1}\n".format(
            tmp, self.hha_transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )
        return fmt_str

    def _check_exists(self) -> bool:
        """
        Only checking for folder existence
        """
        try:
            for split in ["train", "test"]:
                for part, transform in zip(
                    ["rgb", "seg13", "sn", "depth"],
                    [
                        self.rgb_transform,
                        self.seg_transform,
                        self.sn_transform,
                        self.depth_transform,
                    ],
                ):
                    if transform is None:
                        continue
                    path = os.path.join(self.root, f"{split}_{part}")
                    if not os.path.exists(path):
                        raise FileNotFoundError("Missing Folder")
        except FileNotFoundError as e:
            return False
        return True

    def download(self):
        if self._check_exists():
            return
        if self.rgb_transform is not None:
            download_rgb(self.root)
        if self.seg_transform is not None:
            download_seg(self.root)
        if self.sn_transform is not None:
            download_sn(self.root)
        if self.depth_transform is not None or self.hha_transform is not None:
            download_depth(self.root)
        print("Done!")


def download_rgb(root: str):
    train_url = "http://www.doc.ic.ac.uk/~ahanda/nyu_train_rgb.tgz"
    test_url = "http://www.doc.ic.ac.uk/~ahanda/nyu_test_rgb.tgz"

    def _proc(url: str, dst: str):
        if not os.path.exists(dst):
            tar = os.path.join(root, url.split("/")[-1])
            if not os.path.exists(tar):
                download_url(url, root)
            if os.path.exists(tar):
                _unpack(tar)
                _replace_folder(tar.rstrip(".tgz"), dst)
                _rename_files(dst, lambda x: x.split("_")[2])

    _proc(train_url, os.path.join(root, "train_rgb"))
    _proc(test_url, os.path.join(root, "test_rgb"))


def download_seg(root: str):
    train_url = "https://github.com/ankurhanda/nyuv2-meta-data/raw/master/train_labels_13/nyuv2_train_class13.tgz"
    test_url = "https://github.com/ankurhanda/nyuv2-meta-data/raw/master/test_labels_13/nyuv2_test_class13.tgz"

    def _proc(url: str, dst: str):
        if not os.path.exists(dst):
            tar = os.path.join(root, url.split("/")[-1])
            if not os.path.exists(tar):
                download_url(url, root)
            if os.path.exists(tar):
                _unpack(tar)
                _replace_folder(tar.rstrip(".tgz"), dst)
                _rename_files(dst, lambda x: x.split("_")[3])

    _proc(train_url, os.path.join(root, "train_seg13"))
    _proc(test_url, os.path.join(root, "test_seg13"))


def download_sn(root: str):
    url = "https://www.dropbox.com/s/dn5sxhlgml78l03/nyu_normals_gt.zip"
    train_dst = os.path.join(root, "train_sn")
    test_dst = os.path.join(root, "test_sn")

    if not os.path.exists(train_dst) or not os.path.exists(test_dst):
        tar = os.path.join(root, url.split("/")[-1])
        if not os.path.exists(tar):
            req = requests.get(url + "?dl=1")  # dropbox
            with open(tar, "wb") as f:
                f.write(req.content)
        if os.path.exists(tar):
            _unpack(tar)
            if not os.path.exists(train_dst):
                _replace_folder(
                    os.path.join(root, "nyu_normals_gt", "train"), train_dst
                )
                _rename_files(train_dst, lambda x: x[1:])
            if not os.path.exists(test_dst):
                _replace_folder(os.path.join(root, "nyu_normals_gt", "test"), test_dst)
                _rename_files(test_dst, lambda x: x[1:])
            shutil.rmtree(os.path.join(root, "nyu_normals_gt"))


def download_depth(root: str):
    url = "http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat"
    train_dst = os.path.join(root, "train_depth")
    test_dst = os.path.join(root, "test_depth")

    if not os.path.exists(train_dst) or not os.path.exists(test_dst):
        tar = os.path.join(root, url.split("/")[-1])
        if not os.path.exists(tar):
            download_url(url, root)
        if os.path.exists(tar):
            train_ids = [
                f.split(".")[0] for f in os.listdir(os.path.join(root, "train_rgb"))
            ]
            _create_depth_files(tar, root, train_ids)


def _unpack(file: str):
    """
    Unpacks tar and zip, does nothing for any other type
    :param file: path of file
    """
    path = file.rsplit(".", 1)[0]

    if file.endswith(".tgz"):
        tar = tarfile.open(file, "r:gz")
        tar.extractall(path)
        tar.close()
    elif file.endswith(".zip"):
        zip = zipfile.ZipFile(file, "r")
        zip.extractall(path)
        zip.close()


def _rename_files(folder: str, rename_func: callable):
    """
    Renames all files inside a folder based on the passed rename function
    :param folder: path to folder that contains files
    :param rename_func: function renaming filename (not including path) str -> str
    """
    imgs_old = os.listdir(folder)
    imgs_new = [rename_func(file) for file in imgs_old]
    for img_old, img_new in zip(imgs_old, imgs_new):
        shutil.move(os.path.join(folder, img_old), os.path.join(folder, img_new))


def _replace_folder(src: str, dst: str):
    """
    Rename src into dst, replacing/overwriting dst if it exists.
    """
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.move(src, dst)


def _create_depth_files(mat_file: str, root: str, train_ids: list):
    """
    Extract the depth arrays from the mat file into images
    :param mat_file: path to the official labelled dataset .mat file
    :param root: The root directory of the dataset
    :param train_ids: the IDs of the training images as string (for splitting)
    """
    os.mkdir(os.path.join(root, "train_depth"))
    os.mkdir(os.path.join(root, "test_depth"))
    train_ids = set(train_ids)

    depths = h5py.File(mat_file, "r")["depths"]
    for i in range(len(depths)):
        img = (depths[i] * 1e4).astype(np.uint16).T
        id_ = str(i + 1).zfill(4)
        folder = "train" if id_ in train_ids else "test"
        save_path = os.path.join(root, f"{folder}_depth", id_ + ".png")
        Image.fromarray(img).save(save_path)


def get_segmentation_colors(seg: torch.Tensor):
    """
    Converts a segmentation tensor to a colored tensor using a colormap.
    :param seg: A tensor of shape (H, W) or (1, H, W) with integer values in [0, 14]
                representing segmentation classes.
    :return: A tensor of shape (3, H, W) with RGB values in the range [0, 255].
    """
    cmap = plt.cm.get_cmap("tab20", 14)
    colored_seg = cmap(seg.cpu().numpy())
    colored_seg[seg == 0] = [0, 0, 0, 1]  # Set background to black
    colored_seg = colored_seg[:, :, :3]
    colored_seg = torch.from_numpy(colored_seg).permute(2, 0, 1)  # Convert to CxHxW
    return colored_seg