import os
from tqdm import tqdm


root = "data/sunrgbd"
dirs = [
    "depth",
    "image",
    "label13",
    "label37",
]

for dir in dirs:
    dir_path = os.path.join(root, dir)
    subdirs = [d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]
    for subdir in subdirs:
        subdir_path = os.path.join(dir_path, subdir)
        files = sorted(os.listdir(subdir_path))
        print(f"Renaming files in {subdir_path}...")
        for i, file in tqdm(enumerate(files)):
            old_name = os.path.join(subdir_path, file)
            new_name = os.path.join(subdir_path, f"{i:06d}.png")
            os.rename(old_name, new_name)
            print(f"Renamed {old_name} to {new_name}")
        