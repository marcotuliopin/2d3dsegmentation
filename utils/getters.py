import os
import torch
import torch.nn as nn

from models.fcn import get_fcn_resnet50
from models.unet import get_unet
from models.unet_early_fusion_d import get_unet_early_fusion_d
from models.unet_early_fusion_hha_att import get_unet_early_fusion_hha_att
from models.unet_late_fusion_hha_att import get_unet_late_fusion_hha_att
from models.unet_early_fusion_hha import get_unet_early_fusion_hha
from models.unet_late_fusion_d import get_unet_late_fusion_d
from models.unet_late_fusion_hha import get_unet_late_fusion_hha
from models.unet_mid_fusion_hha import get_unet_mid_fusion_hha
from utils.losses import DiceLoss, FocalLoss


# Class weights for the NYUv2 dataset obtained via the inverse log frequency of each class
nyuv2_weights_13 = [0.11756749, 0.58930845, 3.86320268, 1.42978694, 0.61211152,
    0.21107389, 0.14174245, 0.16072167, 1.03913962, 0.87946776,
    0.68799929, 3.74469765, 0.08783193, 0.43534866]



def get_model(name, **kwargs):
    models = {
        "fcn": get_fcn_resnet50,
        "unet": get_unet,
        "unet_early_fusion_d": get_unet_early_fusion_d,
        "unet_early_fusion_hha": get_unet_early_fusion_hha,
        "unet_early_fusion_hha_att": get_unet_early_fusion_hha_att,
        "unet_mid_fusion_hha": get_unet_mid_fusion_hha,
        "unet_late_fusion_d": get_unet_late_fusion_d,
        "unet_late_fusion_hha": get_unet_late_fusion_hha,
        "unet_late_fusion_hha": get_unet_late_fusion_hha_att,
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
            model_params,
            betas=optimizer_config["betas"],
            eps=optimizer_config["eps"],
            lr=optimizer_config["learning_rate"],
            weight_decay=optimizer_config["weight_decay"],
        )
    elif optimizer_name == "sgd":
        return torch.optim.SGD(
            param_groups,
            lr=optimizer_config["learning_rate"],
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
            total_iters=(num_epochs * len_dataloader) // batch_size,
            power=scheduler_config["power"],
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")


def get_latest_checkpoint(checkpoint_dir):
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_dir}")
    return os.path.join(checkpoint_dir, checkpoints[-1])

