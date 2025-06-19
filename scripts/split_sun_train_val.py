from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os


def split_validation_set(root, val_ratio=0.2):
    files = sorted(os.listdir(root))
    if len(files) != 5285:
        raise ValueError(f"Expected 5285 files in {root}, found {len(files)} files.")

    _, val_files = train_test_split(files, test_size=val_ratio, random_state=42)

    if not os.path.exists(root.replace("train", "val")):
        os.makedirs(root.replace("train", "val"))

    for val_file in tqdm(val_files):
        src = os.path.join(root, val_file)
        dst = os.path.join(root.replace("train", "val"), val_file)
        os.rename(src, dst)


print("splitting rgb")
split_validation_set("data/sunrgbd/image/train", 0.2)
print("splitting depth")
split_validation_set("data/sunrgbd/depth/train", 0.2)
print("splitting label37")
split_validation_set("data/sunrgbd/label37/train", 0.2)
