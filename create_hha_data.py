import os
import sys
import cv2
import numpy as np
from tqdm import tqdm

sys.path.append("/home/marcotuliopin/home/studies/poc/hha/")
from hha.getHHA import getHHA
from hha.utils.getCameraParam import getCameraParam

def create_hha_data(source_dir, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    for file in tqdm(sorted(os.listdir(source_dir))):
        print(f"Processing file: {file}")
        if file.endswith('.png'):
            depth_image_path = os.path.join(source_dir, file)
            
            D = cv2.imread(depth_image_path, cv2.IMREAD_ANYDEPTH)
            D = D.astype(np.float32) / 1000.0
            camera_matrix = getCameraParam('color')
            hha_image = getHHA(camera_matrix, D, D)
            
            target_path = os.path.join(target_dir, file)
            cv2.imwrite(target_path, hha_image)
            print(f"Saved HHA image to: {target_path}")

# create_hha_data('data/nyuv2/data/depth/train', 'data/nyuv2/data/hha/train')
create_hha_data('data/nyuv2/data/depth/test', 'data/nyuv2/data/hha/test')