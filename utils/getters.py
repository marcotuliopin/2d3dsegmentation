import os
import numpy as np
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
nyuv2_weights_13 = np.sqrt(nyuv2_weights_13).astype(np.float32).tolist()


nyuv2_weights_40 = [0.30905102, 0.47400618, 0.76580361, 1.01470766, 1.01692224, 1.44338009,
1.60912854, 1.56848911, 2.48611931, 2.76479062, 3.1379106, 0.66807373,
0.5605911, 1.24156069, 2.61150247, 1.04981815, 0.78047966, 1.1479048,
1.44051289, 1.28952995, 2.36916315, 2.80017869, 1.96089424, 1.93057892,
2.2421258, 2.84667172, 1.0108517, 0.95267991, 1.19286297, 1.98246947,
0.96705342, 2.92167463, 1.39214749, 2.25843292, 3.08181635, 0.93553294,
2.69752997, 1.03677536, 1.79582991, 1.50435407]


alpha = [0.09551253169775009, 0.22468185424804688, 0.6091485619544983, 1.021821141242981, 
2.075077533721924, 5.612934112548828, 1.1021182537078857, 0.4463225305080414, 1.0296317338943481, 
0.935192346572876, 1.0341309309005737, 1.3176853656768799, 9.84648323059082, 0.31426239013671875, 
1.074903130531311, 1.6628875732421875, 0.9075990915298462, 0.586455225944519, 1.5414730310440063, 
3.2250051498413086, 1.4229220151901245, 3.930185317993164, 6.180789470672607, 8.53618335723877, 
0.8752219080924988, 2.083346128463745, 7.644067287445068, 2.589294672012329, 3.727135181427002, 
6.819945335388184, 5.100519180297852, 7.276668071746826, 9.497591972351074, 5.027128219604492, 
3.845106363296509, 1.9380747079849243, 2.2630813121795654, 8.103540420532227, 2.460158109664917, 7.841001033782959]


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
            weight=torch.tensor(nyuv2_weights_40).to(device),
            ignore_index=ignore_index,
        )
    elif name == "focal_loss":
        return FocalLoss(
            gamma=loss_config["gamma"],
            alpha=alpha,
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
            T_max=len_dataloader * num_epochs,
            eta_min=scheduler_config["eta_min"],
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")


def get_latest_checkpoint(checkpoint_dir):
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_dir}")
    return os.path.join(checkpoint_dir, checkpoints[-1])

