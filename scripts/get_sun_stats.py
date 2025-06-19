import os
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from collections import defaultdict
from tqdm import tqdm


rgb_root = "data/sunrgbd/image/train"
depth_root = "data/sunrgbd/depth/train"
label_root = "data/sunrgbd/label37/train"


def get_rgb_stats(dir):
    transform = T.ToTensor()
    unique_pixels = [defaultdict(int) for _ in range(3)]

    for file in tqdm(os.listdir(dir)):
        if file.endswith(".png"):
            img_path = os.path.join(dir, file)
            img = Image.open(img_path)
            tensor = transform(img)

            for channel in range(3):
                pixels = tensor[channel].flatten()
                unique_values = np.unique(pixels.numpy(), return_counts=True)
                for i in range(len(unique_values[0])):
                    unique_pixels[channel][unique_values[0][i]] += unique_values[1][i]

    stats = []
    channel_names = ["red", "green", "blue"]

    for channel in range(3):
        pixels = list(unique_pixels[channel].keys())
        counts = list(unique_pixels[channel].values())
        
        mean_val = sum(pixel * count for pixel, count in unique_pixels[channel].items()) / sum(counts)

        variance = sum((pixel - mean_val) ** 2 * count for pixel, count in unique_pixels[channel].items()) / sum(counts)
        std_val = variance ** 0.5

        min_val = min(pixels)
        max_val = max(pixels)

        stats.append(
            {
                "channel": channel_names[channel],
                "mean": mean_val,
                "std": std_val,
                "min": min_val,
                "max": max_val,
            }
        )

        print("--------------------")
        print(f" mean    | {mean_val:.4f}")
        print(f" std     | {std_val:.4f}")
        print(f" min     | {min_val:.4f}")
        print(f" max     | {max_val:.4f}")
        print("--------------------")
        print()

        with open("data/sunrgbd/stats.txt", "a") as f:
            f.write("--------------------\n")
            f.write(f"from {dir}:\n")
            f.write(f"channel {channel + 1}:\n")
            f.write(f" mean    | {mean_val:.4f}\n")
            f.write(f" std     | {std_val:.4f}\n")
            f.write(f" min     | {min_val:.4f}\n")
            f.write(f" max     | {max_val:.4f}\n")
            f.write("--------------------\n\n")

    return stats


def get_1d_stats(dir, transform=None):
    if transform is None:
        transform = T.ToTensor()
    unique_pixels = defaultdict(int)

    for file in tqdm(os.listdir(dir)):
        if file.endswith(".png"):
            img_path = os.path.join(dir, file)
            img = Image.open(img_path)
            tensor = transform(img)

            pixels = tensor.flatten()
            unique_values = np.unique(pixels.numpy(), return_counts=True)
            for i in range(len(unique_values[0])):
                unique_pixels[unique_values[0][i]] += unique_values[1][i]

    pixels = list(unique_pixels.keys())
    counts = list(unique_pixels.values())
    
    mean_val = sum(pixel * count for pixel, count in unique_pixels.items()) / sum(counts)

    variance = sum((pixel - mean_val) ** 2 * count for pixel, count in unique_pixels.items()) / sum(counts)
    std_val = variance ** 0.5

    min_val = min(pixels)
    max_val = max(pixels)

    print("--------------------")
    print(f" mean    | {mean_val:.4f}")
    print(f" std     | {std_val:.4f}")
    print(f" min     | {min_val:.4f}")
    print(f" max     | {max_val:.4f}")
    print("--------------------")
    print()

    with open("data/sunrgbd/stats.txt", "a") as f:
        f.write("--------------------\n")
        f.write(f"from {dir}:\n")
        f.write(f" mean    | {mean_val:.4f}\n")
        f.write(f" std     | {std_val:.4f}\n")
        f.write(f" min     | {min_val:.4f}\n")
        f.write(f" max     | {max_val:.4f}\n")
        f.write("--------------------\n\n")

    return {"mean": mean_val, "std": std_val, "min": min_val, "max": max_val}


class DepthToTensor:
    def __call__(self, depth_img):
        return self._depth_to_tensor(depth_img)

    def _depth_to_tensor(self, depth_img):
        depth_np = np.array(depth_img, dtype=np.uint16)
        depth_meters = depth_np.astype(np.float32) / 1e4 # Convert to meters
        depth_tensor = torch.from_numpy(depth_meters).unsqueeze(0)  # (1, H, W)
        return depth_tensor


print("depth stats:")
get_1d_stats(depth_root, transform=DepthToTensor())
print("label stats:")
get_1d_stats(label_root)
print("rgb stats:")
get_rgb_stats(rgb_root)

