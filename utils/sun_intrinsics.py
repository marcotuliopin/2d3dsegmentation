import numpy as np


def get_sun_intrinsics(device):
    if device == 'xtion':
        fx = 570.342205
        fy = 570.342205
        cx = 310.0
        cy = 225.0
    elif device == 'realsense':
        fx = 691.584229
        fy = 691.584229
        cx = 362.777557
        cy = 264.750000
    elif device == 'kv2':
        fx = 529.500000
        fy = 529.500000
        cx = 365.000000
        cy = 265.000000
    elif device == 'kv1':
        fx = 520.532000
        fy = 520.744400
        cx = 277.925800
        cy = 215.115000
    else:
        raise ValueError(f"Unknown device: {device}")
    return np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0, 0, 1]], dtype=np.float32)