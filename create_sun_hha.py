import os
import cv2
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append("/home/marcotuliopin/home/studies/poc/hha/")
sys.path.append(parent_dir)
from hha.getHHA import getHHA
from utils.sun_intrinsics import get_sun_intrinsics

def create_hha_data(source_dir, target_dir):
    os.makedirs(target_dir, exist_ok=True)

    for file in os.listdir(source_dir):
        print(f"Processing file: {file}")
        if file.endswith('.png'):
            depth_image_path = os.path.join(source_dir, file)
            
            D = cv2.imread(depth_image_path, cv2.COLOR_BGR2GRAY)
            device = get_sun_device(depth_image_path)
            camera_matrix = get_sun_intrinsics(device)
            hha_image = getHHA(camera_matrix, D, D)
            
            target_path = os.path.join(target_dir, file)
            cv2.imwrite(target_path, hha_image)
            print(f"Saved HHA image to: {target_path}")


def get_sun_device(depth_image_path):
    with open('data/sunrgbd/depth/devices.txt', 'r') as f:
        for line in f:
            if depth_image_path in line:
                return line.strip().split()[-1]

create_hha_data('data/sunrgbd/depth/train', 'data/sunrgbd/hha/train')
create_hha_data('data/sunrgbd/depth/test', 'data/sunrgbd/hha/test')
create_hha_data('data/sunrgbd/depth/val', 'data/sunrgbd/hha/val')