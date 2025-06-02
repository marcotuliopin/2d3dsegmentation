import os
import sys
import cv2

sys.path.append("/home/marcotuliopin/home/studies/poc/hha/")
from hha.getHHA import getHHA
from hha.utils.getCameraParam import getCameraParam

def create_hha_data(source_dir, target_dir):
    for file in os.listdir(source_dir):
        print(f"Processing file: {file}")
        if file.endswith('.png'):
            depth_image_path = os.path.join(source_dir, file)
            
            D = cv2.imread(depth_image_path, cv2.COLOR_BGR2GRAY) / 10000.0
            camera_matrix = getCameraParam('color')
            hha_image = getHHA(camera_matrix, D, D)
            
            target_path = os.path.join(target_dir, file.replace('.png', '_hha.png'))
            cv2.imwrite(target_path, hha_image)
            print(f"Saved HHA image to: {target_path}")

create_hha_data('data/nyuv2/train_depth', 'data/nyuv2/train_hha')
create_hha_data('data/nyuv2/test_depth', 'data/nyuv2/test_hha')