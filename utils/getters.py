import os
import numpy as np
import torch
import torch.nn as nn

from models.late_fusion import LateFusion
from models.mid_fusion_att import AttentionMidFusion
from models.rgb_only import RBGOnly
from models.early_fusion_d import EarlyFusionD
from models.early_fusion_hha_att import AttentionEarlyFusionHHA
from models.late_fusion_att import AttentionLateFusion
from models.early_fusion_hha import EarlyFusionHHA
from utils.losses import DiceLoss, FocalLoss


# Class weights for the NYUv2 dataset obtained via the inverse log frequency of each class
nyuv2_weights_13 = [0.11756749, 0.58930845, 3.86320268, 1.42978694, 0.61211152,
    0.21107389, 0.14174245, 0.16072167, 1.03913962, 0.87946776,
    0.68799929, 3.74469765, 0.08783193, 0.43534866]
nyuv2_weights_13 = np.sqrt(nyuv2_weights_13).astype(np.float32)

nyuv2_weights_40 = [0.3362954, 0.51579217, 0.833313, 1.10415918, 1.10656899, 1.57062122,
1.75098122, 1.70675922, 2.70528308, 3.00852065, 3.41453301, 0.72696774,
0.37219925, 0.61000999, 1.35101043, 2.84171938, 1.14236484, 0.84928282,
1.24909831, 1.56750125, 1.40320842, 2.57801666, 3.04702835, 2.13375681,
2.10076904, 2.43978033, 3.09761998, 1.0999633, 1.03666337, 1.29801976,
2.157234, 1.05230399, 3.17923476, 1.5148722, 2.457525, 3.35349377,
1.01800481, 2.93533064, 1.12817226, 1.95414124, 1.63697035]


def get_model(name, **kwargs):
    models = {
        "rgb_only": RBGOnly,
        "early_fusion_d": EarlyFusionD,
        "early_fusion_hha": EarlyFusionHHA,
        "early_fusion_hha_att": AttentionEarlyFusionHHA,
        "mid_fusion_hha_att": AttentionMidFusion,
        "late_fusion_d": LateFusion,
        "late_fusion_hha": LateFusion,
        "late_fusion_hha_att": AttentionLateFusion,
    }
    
    if name not in models:
        raise ValueError(f"Model {name} not supported. Options are: {list(models.keys())}")
    
    return models[name](**kwargs)


def get_loss_function(name: str, loss_config: dict, ignore_index: int, device: str):
    if name == "cross_entropy":
        return nn.CrossEntropyLoss()
    elif name == "weighted_cross_entropy":
        return nn.CrossEntropyLoss(
            weight=torch.tensor(nyuv2_weights_13).to(device),
            ignore_index=ignore_index,
        )
    elif name == "focal_loss":
        return FocalLoss(
            gamma=loss_config["gamma"],
            ignore_index=ignore_index,
            reduction=loss_config["reduction"]
        )
    elif name == "dice_loss":
        return DiceLoss(
            smooth=loss_config["smooth"],
            ignore_index=ignore_index,
        )
    else:
        raise ValueError(f"Unknown loss function: {name}")


def get_optimizer(optimizer_name: str, model_params: dict, optimizer_config: dict, param_groups: list):
    if optimizer_name == "adam":
        return torch.optim.AdamW(
            param_groups,
            weight_decay=optimizer_config["weight_decay"],
        )
    elif optimizer_name == "sgd":
        return torch.optim.SGD(
            param_groups,
            momentum=optimizer_config["momentum"],
            weight_decay=optimizer_config["weight_decay"],
            nesterov=optimizer_config["nesterov"]
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def get_scheduler(scheduler_name, optimizer, scheduler_config, batch_size, num_epochs, len_dataloader):
    if scheduler_name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_config["step_size"],
            gamma=scheduler_config["gamma"],
        )
    elif scheduler_name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=scheduler_config["factor"],
            patience=scheduler_config["patience"],
        )
    elif scheduler_name == "polynomial":
        return torch.optim.lr_scheduler.PolynomialLR(
            optimizer,
            total_iters=num_epochs,
            power=scheduler_config["power"],
        )
    elif scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
            eta_min=scheduler_config["eta_min"],
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")


def get_latest_checkpoint(checkpoint_dir):
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_dir}")
    return os.path.join(checkpoint_dir, checkpoints[-1])

