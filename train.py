import datetime
import os
import pickle
import argparse
import random
from time import time
import numpy as np
import yaml
import torch

from utils.dataloader import nyuv2_dataloader
from utils.getters import get_loss_function, get_model, get_optimizer, get_scheduler
from utils.runner import Runner
from utils.training import CheckpointSaver
from torchinfo import summary


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
    parser.add_argument(
        "--scheduler",
        type=str,
        choices=[key for key in config["train"]["lr_scheduler"]],
        help="Scheduler to be used",
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
    exp_dir = os.path.join(config["output"]["directories"]["checkpoints"], args.experiment_name)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(config["output"]["directories"]["plots"], args.experiment_name), exist_ok=True)

    # ------ Data  Loading ------
    train_loader, val_loader = nyuv2_dataloader(
        train=True,
        split_val=True,
        seed=args.seed,
        batch_size=args.batch_size,
        rgb_only=config["model"][args.model]["rgb_only"],
        use_hha=config["model"][args.model]["use_hha"],
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
    with open(os.path.join(exp_dir, "model_summary.txt"), "w") as f:
        f.write(str(summary(model, (1, 6, 240, 360))))

    optimizer_params = model.get_optimizer_groups()

    # ------ Training Configurations ------
    criterion = get_loss_function(
        args.loss,
        config["train"]["loss"][args.loss],
        config["data"]["unlabeled_id"],
        device=device,
    )
    optimizer = get_optimizer(
        args.optimizer,
        model.parameters(),
        config["train"]["optimizer"][args.optimizer],
        optimizer_params,
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

    start_epoch = 0
    if args.resume:
        checkpoint_path = os.path.join(exp_dir, "checkpoint.pth")
        start_epoch = checkpoint_saver.load(checkpoint_path)

    trainer = Runner(
        model,
        device,
        optimizer,
        criterion,
        config["model"][args.model]["rgb_only"],
        config["model"][args.model]["use_hha"],
    )

    # ------ Training ------
    best_miou = 0
    non_improved_epochs = 0
    
    start_time = time()
    for epoch in range(start_epoch, args.epochs):
        train_loss = trainer.train_epoch(train_loader)
        val_loss, val_miou = trainer.validate(val_loader)

        scheduler.step()

        if val_miou > best_miou + config["train"]["early_stopping"]["delta"]:
            print(f"Validation mIoU improved from {best_miou:.4f} to {val_miou:.4f}. Saving model.")
            checkpoint_saver.save(epoch, verbose=True)
            best_miou = val_miou
            non_improved_epochs = 0
        else:
            non_improved_epochs += 1
            if non_improved_epochs >= config["train"]["early_stopping"]["patience"]:
                print(f"Early stopping at epoch {epoch} due to no improvement.")
                break

        print_log(
            experiment_name=args.experiment_name,
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            val_miou=val_miou,
            learning_rate=scheduler.get_last_lr()[0],
            time_elapsed=time() - start_time,
            non_improved_epochs=non_improved_epochs,
        )

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


def print_log(
    experiment_name,
    epoch,
    train_loss,
    val_loss,
    val_miou,
    learning_rate,
    time_elapsed,
    non_improved_epochs,
):
    print("----------------------------------")
    print(f"| experiment: {experiment_name:<30}")
    print(f"| time/               |          |")
    print(f"|    epochs           | {epoch + 1:<8} |")
    print(f"|    time_elapsed     | {time_elapsed:<8.1f} |")
    print(f"| train/              |          |")
    print(f"|    non_improved     | {non_improved_epochs:<8} |")
    print(f"|    learning_rate    | {learning_rate:<8.6f} |")
    print(f"| metrics/            |          |")
    print(f"|    train_loss       | {train_loss:<8.4f} |")
    print(f"|    val_loss         | {val_loss:<8.4f} |")
    print(f"|    val_miou         | {val_miou:<8.4f} |")
    print("----------------------------------")


if __name__ == "__main__":
    config = read_config()
    args = parse_args(config)
    save_experiment_config(config, args)
    main(args, config)
