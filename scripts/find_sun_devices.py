# xtion intrinsics:
# 570.342205 0.000000 310.000000
# 0.000000 570.342205 225.000000
# 0.000000 0.000000 1.000000

# realsense intrinsics:
# 691.584229 0.000000 362.777557
# 0.000000 691.584229 264.750000
# 0.000000 0.000000 1.000000

# kv2 intrinsics:
# 529.500000 0.000000 365.000000
# 0.000000 529.500000 265.000000
# 0.000000 0.000000 1.000000

# kv1 intrinsics:
# 520.532000 0.000000 277.925800
# 0.000000 520.744400 215.115000
# 0.000000 0.000000 1.000000

import os
from PIL import Image


device_img_sizes = {
    "xtion": (591, 441),
    "realsense": (681, 531),
    "kv2": (730, 530),
    "kv1": (561, 427),
}

root = "data/sunrgbd/depth"

dirs = [root + "/test", root + "/train", root + "/val"]

for d in dirs:
    for file in os.listdir(d):
        if file.endswith(".png"):
            filepath = os.path.join(d, file)
            img = Image.open(filepath)
            if img.size not in device_img_sizes.values():
                print(f"Image {file} has unexpected size: {img.size}")
                break
            for device, size in device_img_sizes.items():
                if img.size == size:
                    with open("data/sunrgbd/depth/devices.txt", "a") as f:
                        f.write(f"{filepath} {device}\n")
