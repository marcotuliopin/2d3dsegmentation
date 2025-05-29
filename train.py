import datetime
import os
import pickle
import argparse
import random
import numpy as np
import yaml
import torch
import torch.nn as nn

from models.deeplabv3_resnet101 import get_deeplabv3_resnet101
from models.deeplabv3_resnet50 import get_deeplabv3_resnet50
from models.dual_encoder_unet import get_dual_encoder_unet
from models.fcn_resnet101 import get_fcn_resnet101
from models.fcn_resnet50 import get_fcn_resnet50
from models.unet import get_unet
from utils.dataloader import nyuv2_dataloader
from utils.training import (
    CheckpointSaver,
    EarlyStopping,
    Trainer,
)


def parse_args(config):
    parser = argparse.ArgumentParser(description="Segmentation Training Script")
    parser.add_argument(
        "--model",
        type=str,
        choices=[key for key in config["model"].keys()],
        help="Model to be used",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=config["train"]["batch_size"],
        help="Batch size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=config["train"]["epochs"],
        help="Number of epochs",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=[key for key in config["train"]["optimizer"]],
        help="Optimizer to be used",
    )
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument(
        "--scheduler",
        type=str,
        choices=[key for key in config["train"]["lr_scheduler"]],
        help="Scheduler to be used",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=config["train"]["early_stopping"]["patience"],
        help="Patience for early stopping",
    )
    parser.add_argument(
        "--loss",
        type=str,
        choices=[key for key in config["train"]["loss"]],
        help="Loss function to be used",
    )
    parser.add_argument(
        "-n",
        "--experiment-name",
        type=str,
        default="experiment",
        help="Experiment name",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the latest checkpoint",
    )
    # TODO: Add seed in every random operation
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


def main(args, config):
    # Configure CUDA
    device = configure_device()
    set_seed(args.seed)

    # Make sure the output directories exist
    exp_dir = os.path.join(
        config["output"]["directories"]["checkpoints"], args.experiment_name
    )
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(
        os.path.join(config["output"]["directories"]["plots"], args.experiment_name),
        exist_ok=True,
    )

    # ------ Data  Loading ------
    train_loader, val_loader = nyuv2_dataloader(
        train=True,
        split_val=True,
        rgb_only=config["model"][args.model]["rgb_only"],
        batch_size=args.batch_size,
        num_workers=config["train"]["num_workers"],
        image_size=config["data"]["shape"],
    )

    # ------ Model Configuration ------
    model = get_model(
        num_classes=config["data"]["num_classes"],
        name=config["model"][args.model]["name"],
        **config["model"][args.model]["config"],
    )
    model = model.to(device)

    # ------ Training Configurations ------
    # TODO: Fix weight calculation
    criterion = get_loss_function(
        args.loss,
        config["train"]["loss"][args.loss],
        config["data"]["unlabeled_id"],
    )
    optimizer = get_optimizer(
        args.optimizer,
        model.parameters(),
        config["train"]["optimizer"][args.optimizer],
        args.lr,
    )
    scheduler = get_scheduler(
        args.scheduler,
        optimizer,
        config["train"]["lr_scheduler"][args.scheduler],
        args.batch_size,
        args.epochs,
        len(train_loader),
    )
    checkpoint_saver = CheckpointSaver(model, exp_dir, optimizer)
    early_stopping = EarlyStopping(
        patience=config["train"]["early_stopping"]["patience"],
        min_delta=config["train"]["early_stopping"]["delta"],
        checkpoint_saver=checkpoint_saver,
    )

    start_epoch = 0
    if args.resume:
        checkpoint_path = os.path.join(exp_dir, "checkpoint.pth")
        start_epoch = checkpoint_saver.load(checkpoint_path)

    trainer = Trainer(model, optimizer, criterion, device)

    # ------ Training ------
    for epoch in range(start_epoch, args.epochs):
        train_loss = trainer.train_epoch(train_loader, epoch)
        val_loss = trainer.validate(val_loader, epoch)

        scheduler.step(val_loss)

        if early_stopping(val_loss, epoch):
            break

        print(f"End of epoch {epoch} of experiment {args.experiment_name}:")
        print(f"  train loss: {train_loss:.4f}")
        print(f"  val loss: {val_loss:.4f}")

    print(f"Training complete. Results saved in {exp_dir}")


def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def configure_device():
    print("CUDA Available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print(
            f"GPU Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )
        print(f"GPU Allocated Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"GPU Reserved Memory: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        print("CUDA Version:", torch.version.cuda)
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def get_model(name, **kwargs):
    models = {
        "fcn_resnet50": get_fcn_resnet50,
        "deeplabv3_resnet50": get_deeplabv3_resnet50,
        "fcn_resnet101": get_fcn_resnet101,
        "deeplabv3_resnet101": get_deeplabv3_resnet101,
        "unet": get_unet,
        "dual_encoder_unet": get_dual_encoder_unet,
    }
    
    if name not in models:
        raise ValueError(f"Model {name} não suportado. Opções: {list(models.keys())}")
    
    return models[name](**kwargs)


def get_loss_function(name: str, loss_config: dict, ignore_index: int):
    if name == "cross_entropy":
        return nn.CrossEntropyLoss()
    # elif loss_name == "weighted_cross_entropy":
    #     return nn.CrossEntropyLoss(
    #         weight=weight,
    #         ignore_index=ignore_index,
    #     )
    elif name == "focal_loss":
        from utils.losses import FocalLoss

        return FocalLoss(
            alpha=loss_config["alpha"],
            gamma=loss_config["gamma"],
            ignore_index=ignore_index,
        )
    elif name == "dice_loss":
        from utils.losses import DiceLoss

        return DiceLoss(
            smooth=loss_config["smooth"],
            ignore_index=ignore_index,
        )
    else:
        raise ValueError(f"Unknown loss function: {name}")


def get_optimizer(optimizer_name: str, model_params: dict, optimizer_config: dict, lr: float):
    if optimizer_name == "adam":
        return torch.optim.AdamW(
            model_params,
            betas=optimizer_config["betas"],
            eps=optimizer_config["eps"],
            lr=lr,
            weight_decay=optimizer_config["weight_decay"],
        )
    elif optimizer_name == "sgd":
        return torch.optim.SGD(
            model_params,
            lr=lr,
            momentum=optimizer_config["momentum"],
            weight_decay=optimizer_config["weight_decay"],
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


def read_config():
    with open("config.yml", "r") as file:
        config = yaml.safe_load(file)
    return config


def save_experiment_config(config, args):
    exp_dir = os.path.join(
        config["output"]["directories"]["checkpoints"], args.experiment_name
    )
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(
        os.path.join(config["output"]["directories"]["plots"], args.experiment_name),
        exist_ok=True,
    )

    experiment_config = {
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "num_classes": config["data"]["num_classes"],
        "model": config["model"][args.model],
        "loss": args.loss,
        "optimizer": args.optimizer,
        "optimizer_config": config["train"]["optimizer"][args.optimizer],
        "scheduler": args.scheduler,
        "scheduler_config": config["train"]["lr_scheduler"][args.scheduler],
        "shape": config["data"]["shape"],
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    }

    with open(os.path.join(exp_dir, "config.pkl"), "wb") as f:
        pickle.dump(experiment_config, f)

    def sanitize_config(item):
        if isinstance(item, dict):
            return {k: sanitize_config(v) for k, v in item.items()}
        elif isinstance(item, list):
            return [sanitize_config(v) for v in item]
        elif isinstance(item, (int, float, str, bool, type(None))):
            return item
        else:
            return str(item)

    experiment_config = sanitize_config(experiment_config)

    config_path = os.path.join(exp_dir, f"{args.experiment_name}_config.yml")
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(experiment_config, f, default_flow_style=False, encoding=None)


if __name__ == "__main__":
    config = read_config()
    args = parse_args(config)
    save_experiment_config(config, args)
    main(args, config)
