import os

ROOT_DIR = "data/SUN_RGBD/"

DATA_CONFIG = {
    "root_dir": ROOT_DIR,
    "train_file": os.path.join(ROOT_DIR, "train13.txt"),
    "val_file": os.path.join(ROOT_DIR, "split/val13.txt"),
    "test_file": os.path.join(ROOT_DIR, "split/test13.txt"),
    "num_classes": 14,
    "image_height": 256,
    "image_width": 352,
}

TRAINING_CONFIG = {
    "batch_size": 4,
    "num_workers": 4,
    "lr": 1e-4,
    "weight_decay": 1e-3,
    "num_epochs": 100,
    "patience": 10,
    "min_delta": 5e-4,
    "device": "cuda",
}

MODEL_CONFIGS = {
    "fcn_resnet50": {
        "pretrained": False,
        "dropout": 0.5,
    },
    "fcn_resnet101": {
        "pretrained": False,
        "dropout": 0.5,
    },
    "deeplabv3_resnet50": {
        "pretrained": False,
        "dropout": 0.5,
    },
    "deeplabv3_resnet101": {
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