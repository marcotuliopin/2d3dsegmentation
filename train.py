import datetime
import os
import pickle
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.datasets import NYUDepthV2Dataset, calculate_class_weights
from utils.model import get_model
from utils.transforms import get_training_transforms, get_validation_transforms
from utils.training import (
    CheckpointSaver,
    EarlyStopping,
    Trainer,
)
from utils.visualization import plot_loss_curves


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
        help="Resume training from the latest checkpoint"
    )
    # TODO: Add seed in every random operation
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze the backbone during training",
    )
    parser.add_argument(
        "--use-depth",
        action="store_true",
        help="Use depth as a fourth channel in the input",
    )
    return parser.parse_args()


def main(args, config):
    # Configure CUDA
    device = configure_device()

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
    data_config = config["data"][args.data].copy()

    train_transform = get_training_transforms(height=data_config["image_size"][0], width=data_config["image_size"][1])
    val_transform = get_validation_transforms(height=data_config["image_size"][0], width=data_config["image_size"][1])

    train_loader = DataLoader(
        NYUDepthV2Dataset(
            path_file=data_config["paths"]["train_file"],
            transform=train_transform,
            use_depth=args.use_depth,
        ),
        batch_size=args.batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=config["train"]["num_workers"],
    )
    val_loader = DataLoader(
        NYUDepthV2Dataset(
            path_file=data_config["paths"]["val_file"],
            transform=val_transform,
            use_depth=args.use_depth,
        ),
        batch_size=args.batch_size,
        drop_last=True,
        shuffle=False,
        num_workers=config["train"]["num_workers"],
    )

    # ------ Model Configuration ------
    in_channels = 4 if args.use_depth else 3
    model_config = config["model"]["common"].copy()
    model_config.update(config["model"][args.model])
    model = get_model(args.model, num_classes=data_config["num_classes"], in_channels=in_channels, **model_config)
    model = model.to(device)

    # ------ Training Configurations ------
    weight = calculate_class_weights(train_loader, data_config["num_classes"], data_config["unlabeled_id"])
    criterion = get_loss_function(args.loss, config["train"]["loss"][args.loss], data_config["unlabeled_id"], weight.to(device))
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
        start_epoch = load_checkpoint(checkpoint_saver, exp_dir)

    trainer = Trainer(model, optimizer, criterion, device)

    train_losses = []
    val_losses = []

    # ------ Training ------
    for epoch in range(start_epoch, args.epochs):
        train_loss = trainer.train_epoch(train_loader, epoch)
        val_loss = trainer.validate(val_loader, epoch)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if args.scheduler == "plateau":
            scheduler.step(val_loss)
        else:
            scheduler.step()

        print(f"End of epoch {epoch} of experiment {args.experiment_name}:")
        print(f"  train loss: {train_loss:.4f}")
        print(f"  val loss: {val_loss:.4f}")

        if early_stopping(val_loss, epoch):
            break

    # ------ Saving Info ------
    plot_path = os.path.join(
        config["output"]["directories"]["plots"],
        args.experiment_name,
        "loss_curves.png",
    )
    plot_loss_curves(train_losses, val_losses, save_path=plot_path)

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


def get_loss_function(loss_name, loss_config, ignore_index, weight):
    if loss_name == "cross_entropy":
        return nn.CrossEntropyLoss(
            ignore_index=ignore_index,
        )
    elif loss_name == "weighted_cross_entropy":
        return nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_index,
        )
    elif loss_name == "focal_loss":
        from utils.losses import FocalLoss
        return FocalLoss(
            alpha=loss_config["alpha"],
            gamma=loss_config["gamma"],
            ignore_index=ignore_index,
        )
    elif loss_name == "dice_loss":
        from utils.losses import DiceLoss
        return DiceLoss(
            smooth=loss_config["smooth"],
            ignore_index=ignore_index,
        )
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")


def get_optimizer(optimizer_name, model_params, optimizer_config, lr):
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


def load_checkpoint(checkpoint_saver, exp_dir):
    start_epoch = 0
    checkpoint_path = os.path.join(exp_dir, "checkpoint.pth")
    if os.path.exists(checkpoint_path):
        start_epoch = checkpoint_saver.load(checkpoint_path)
        print(f"Resuming training from epoch {start_epoch}")
    else:
        print(f"No checkpoint found at {checkpoint_path}, starting training from scratch.")

    return start_epoch


def read_config():
    with open("config.yml", "r") as file:
        config = yaml.safe_load(file)
    return config


def save_experiment_config(config, args):
    exp_dir = os.path.join(config["output"]["directories"]["checkpoints"], args.experiment_name)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(
        os.path.join(config["output"]["directories"]["plots"], args.experiment_name),
        exist_ok=True,
    )

    experiment_config = {
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "num_classes": config["data"][args.data]["num_classes"],
        "model": args.model,
        "model_config": config["model"][args.model],
        "loss": args.loss,
        "optimizer": args.optimizer,
        "optimizer_config": config["train"]["optimizer"][args.optimizer],
        "scheduler": args.scheduler,
        "scheduler_config": config["train"]["lr_scheduler"][args.scheduler],
        "freeze_backbone": args.freeze_backbone,
        "use_depth": args.use_depth,
        "image_size": config["data"][args.data]["image_size"],
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
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(experiment_config, f, default_flow_style=False, encoding=None)


if __name__ == "__main__":
    config = read_config()
    args = parse_args(config)
    save_experiment_config(config, args)
    main(args, config)
