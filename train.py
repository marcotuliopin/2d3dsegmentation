import os
import pickle
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import yaml

from models.segmentation import get_model
from utils.datasets import NYUDepthV2Dataset
from utils.transforms import get_training_transforms, get_validation_transforms
from utils.training import (
    CheckpointSaver,
    EarlyStopping,
    Trainer,
)
from utils.visualization import plot_loss_curves


def main(args):
    # Configure CUDA
    device = configure_device()

    # Make sure the output directories exist
    exp_dir = os.path.join(config["output"]["directories"]["checkpoints"], args.experiment_name)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(config["output"]["directories"]["plots"], args.experiment_name), exist_ok=True)

    # ------ Data  Loading ------
    data_config = config["data"][args.data].copy()
    train_transform = get_training_transforms(
        height=data_config["image_size"][0], width=data_config["image_size"][1]
    )
    val_transform = get_validation_transforms(
        height=data_config["image_height"][0], width=data_config["image_width"][1]
    )

    full_dataset = NYUDepthV2Dataset(
        path_file=data_config["paths"]["train_file"], transform=None
    )
    indices = list(range(len(full_dataset)))
    train_indices, val_indices = train_test_split(
        indices, test_size=0.2, random_state=config["train"]["seed"], shuffle=True
    )

    train_dataset = Subset(
        NYUDepthV2Dataset(data_config["train_file"], transform=train_transform),
        train_indices,
    )
    val_dataset = Subset(
        NYUDepthV2Dataset(data_config["train_file"], transform=val_transform),
        val_indices,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=config["train"]["num_workers"]["num_workers"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        drop_last=True,
        shuffle=False,
        num_workers=config["train"]["num_workers"],
    )

    # ------ Model Configuration ------
    model_config = config["model"]["common"].copy()
    model_config.update(config["model"][args.model])
    model = get_model(
        args.model, num_classes=data_config["num_classes"], **model_config
    )
    model = model.to(device)

    # ------ Training Configurations ------
    criterion = get_loss_function(args.loss, config["train"]["loss"], data_config["unlabeled_id"])
    optimizer = get_optimizer(args.optimizer, model.parameters(), config["train"]["optimizer"], args.lr)
    scheduler = get_scheduler(args.scheduler, optimizer, config["train"]["lr_scheduler"])
    checkpoint_saver = CheckpointSaver(model, exp_dir, optimizer)
    early_stopping = EarlyStopping(
        patience=config["train"]["early_stopping"]["patience"],
        min_delta=config["train"]["early_stopping"]["delta"],
        checkpoint_saver=checkpoint_saver,
    )

    trainer = Trainer(model, optimizer, criterion, device)

    train_losses = []
    val_losses = []

    # ------ Training ------
    for epoch in range(args.epochs):
        train_loss = trainer.train_epoch(train_loader, epoch)
        val_loss = trainer.validate(val_loader, epoch)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        loss_path = os.path.join(exp_dir, f"{args.experiment_name}_losses.pkl")
        with open(loss_path, "wb") as f:
            pickle.dump({"train": train_losses, "val": val_losses}, f)

        print(f"End of epoch {epoch}:")
        print(f"  train loss: {train_loss:.4f}")
        print(f"  val loss: {val_loss:.4f}")

        if early_stopping(val_loss, epoch):
            break

    # ------ Saving Info ------
    plot_path = os.path.join(
        config["output"]["directories"]["plots"], args.experiment_name, "loss_curves.png"
    )
    plot_loss_curves(train_losses, val_losses, save_path=plot_path)

    # Save experiment config
    config = {
        "model": args.model,
        "pretrained": args.pretrained,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "num_classes": data_config["num_classes"],
        "model_config": model_config,
    }

    with open(os.path.join(exp_dir, "config.pkl"), "wb") as f:
        pickle.dump(config, f)

    print(f"Training complete. Results saved in {exp_dir}")


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


def get_loss_function(loss_name, loss_config, ignore_index):
    if loss_name == "cross_entropy":
        return nn.CrossEntropyLoss(
            weight=loss_config["weight"],
            ignore_index=ignore_index,
        )
    elif loss_name == "focal_loss":
        from utils.training import FocalLoss
        return FocalLoss(
            alpha=loss_config["alpha"],
            gamma=loss_config["gamma"],
            ignore_index=ignore_index,
        )
    elif loss_name == "dice_loss":
        from utils.training import DiceLoss
        return DiceLoss(
            smooth=loss_config["smooth"],
            ignore_index=ignore_index,
        )
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")


def get_optimizer(optimizer_name, model_params, optimizer_config, lr):
    if optimizer_name == "adam":
        return torch.optim.Adam(
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


def get_scheduler(scheduler_name, optimizer, scheduler_config):
    if scheduler_name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_config[scheduler_name]["step_size"],
            gamma=scheduler_config[scheduler_name]["gamma"],
        )
    elif scheduler_name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=scheduler_config[scheduler_name]["factor"],
            patience=scheduler_config[scheduler_name]["patience"],
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")


def read_config():
    with open("config.yml", "r") as file:
        config = yaml.safe_load(file)
    return config


def parse_args(config):
    parser = argparse.ArgumentParser(description="Segmentation Training Script")
    parser.add_argument(
        "--model",
        type=str,
        choices=[key for key in config["model"].keys() if key != "common"],
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
        "--data",
        type=str,
        default="nyu_depth_v2",
        choices=[key for key in config["data"].keys()],
        help="Dataset to be used",
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
        choices=[key for key in config["train"]["scheduler"]],
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
        "-n", "--experiment_name", type=str, default="experiment", help="Experiment name"
    )
    return parser.parse_args()


if __name__ == "__main__":
    config = read_config()
    args = parse_args(config)
    main(args)
