import os
import torch
from matplotlib import pyplot as plt
from PIL import Image
from torch.utils.data import Dataset


class NYUv2(Dataset):
    def __init__(
        self,
        root: str,
        train: bool = True,
        rgb_transform=None,
        seg_transform=None,
        sn_transform=None,
        depth_transform=None,
        hha_transform=None,
        sync_transform=None,
    ):
        super().__init__()
        self.root = root

        self.rgb_transform = rgb_transform
        self.seg_transform = seg_transform
        self.sn_transform = sn_transform
        self.depth_transform = depth_transform
        self.hha_transform = hha_transform
        self.sync_transform = sync_transform

        self.train = train
        self._split = "train" if train else "test"

        self._files = sorted(os.listdir(os.path.join(root, f"image/{self._split}")))

    def __getitem__(self, index: int):
        folder = lambda name: os.path.join(self.root, f"{name}/{self._split}")
        imgs = []

        if self.rgb_transform is not None:
            img = Image.open(os.path.join(folder("image"), self._files[index]))
            imgs.append(img)

        if self.seg_transform is not None:
            img = Image.open(os.path.join(folder("seg40"), self._files[index]))
            imgs.append(img)

        if self.depth_transform is not None:
            img = Image.open(os.path.join(folder("depth"), self._files[index]))
            imgs.append(img)
        
        if self.hha_transform is not None:
            file_index = self._files[index].split('.')[0][1:]
            hha_filename = f"{file_index}_hha.png"
            img = Image.open(os.path.join(folder("hha"), hha_filename))
            imgs.append(img)
        
        imgs = self._augment(imgs) 

        return imgs

    def __len__(self):
        return len(self._files)
    
    def _augment(self, imgs):
        if self.sync_transform is not None:
            if len(imgs) > 2:
                imgs = self.sync_transform(
                    imgs[0],
                    imgs[1],
                    imgs[2] if self.depth_transform is not None else None,
                    imgs[2] if self.hha_transform is not None else None,
                )
            else:
                imgs = self.sync_transform(imgs[0], imgs[1])

        if self.rgb_transform is not None:
            imgs[0] = self.rgb_transform(imgs[0])
        if self.seg_transform is not None:
            imgs[1] = self.seg_transform(imgs[1])
        if self.depth_transform is not None:
            imgs[2] = self.depth_transform(imgs[2])
        if self.hha_transform is not None:
            imgs[2] = self.hha_transform(imgs[2])

        return imgs


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