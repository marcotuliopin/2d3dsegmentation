import os

ROOT_DIR = {
    "sun_rgbd": "data/SUN_RGBD",
    "nyu_depth_v2": "data/NYUDepthv2_seg/data"
}

DATA_CONFIG = {
    "sun_rgbd": {
        "root_dir": ROOT_DIR["sun_rgbd"],
        "train_file": os.path.join(ROOT_DIR["sun_rgbd"], "train37.txt"),
        "val_file": os.path.join(ROOT_DIR["sun_rgbd"], "split/val37.txt"),
        "test_file": os.path.join(ROOT_DIR["sun_rgbd"], "split/test37.txt"),
        "unlabeled": 0,
        "num_classes": 38,
        "image_height": 256,
        "image_width": 256,
    },
    "nyu_depth_v2": {
        "root_dir": ROOT_DIR["nyu_depth_v2"],
        "train_file": os.path.join(ROOT_DIR["nyu_depth_v2"], "train.parquet"),
        "test_file": os.path.join(ROOT_DIR["nyu_depth_v2"], "test.parquet"),
        "unlabeled": 40,
        "num_classes": 41,
        "image_height": 224,
        "image_width": 224,
    }
}

TRAINING_CONFIG = {
    "batch_size": 4,
    "num_workers": 4,
    "lr": 5e-5,
    "weight_decay": 1e-3,
    "num_epochs": 100,
    "patience": 7,
    "min_delta": 5e-4,
    "device": "cuda",
}

MODEL_CONFIGS = {
    "fcn_resnet50": {
        "pretrained": False,
        "dropout": 0.2,
    },
    "fcn_resnet101": {
        "pretrained": False,
        "dropout": 0.2,
    },
    "deeplabv3_resnet50": {
        "pretrained": False,
        "dropout": 0.2,
    },
    "deeplabv3_resnet101": {
        "pretrained": False,
        "dropout": 0.2,
    },
    "unet_resnet50": {
        "pretrained": False,
        "dropout": 0.5,
    },
}

OUTPUT_CONFIG = {
    "checkpoints_dir": "checkpoints",
    "results_dir": "results",
    "plots_dir": "plots",
}

for directory in OUTPUT_CONFIG.values():
    os.makedirs(directory, exist_ok=True)
