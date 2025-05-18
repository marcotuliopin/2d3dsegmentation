import os
import torch
import torch.nn as nn
import pickle
import argparse
from torch.utils.data import DataLoader

from config import DATA_CONFIG, TRAINING_CONFIG, MODEL_CONFIGS, OUTPUT_CONFIG
from models.segmentation import get_model
from utils.datasets import SunRGBDDataset
from utils.transforms import get_training_transforms, get_validation_transforms
from utils.training import (
    CheckpointSaver,
    EarlyStopping,
    FocalLoss,
    SegmentationTrainer,
)
from utils.visualization import plot_loss_curves


def parse_args():
    parser = argparse.ArgumentParser(description="Treinamento de modelo de segmentação")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="fcn_resnet50",
        choices=[
            "fcn_resnet50",
            "deeplabv3_resnet50",
            "fcn_resnet101",
            "deeplabv3_resnet101",
        ],
        help="Modelo a ser treinado",
    )
    parser.add_argument(
        "-p", "--pretrained", action="store_true", help="Usar pesos pré-treinados"
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=TRAINING_CONFIG["batch_size"],
        help="Tamanho do batch",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=TRAINING_CONFIG["num_epochs"],
        help="Número de épocas",
    )
    parser.add_argument(
        "--lr", type=float, default=TRAINING_CONFIG["lr"], help="Taxa de aprendizado"
    )
    parser.add_argument(
        "-n", "--exp-name", type=str, default="experiment", help="Nome do experimento"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    exp_dir = os.path.join(OUTPUT_CONFIG["checkpoints_dir"], args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_CONFIG["plots_dir"], args.exp_name), exist_ok=True)

    train_transform = get_training_transforms(
        height=DATA_CONFIG["image_height"], width=DATA_CONFIG["image_width"]
    )
    val_transform = get_validation_transforms(
        height=DATA_CONFIG["image_height"], width=DATA_CONFIG["image_width"]
    )

    train_dataset = SunRGBDDataset(
        path_file=DATA_CONFIG["train_file"],
        root_dir=DATA_CONFIG["root_dir"],
        transform=train_transform,
    )

    val_dataset = SunRGBDDataset(
        path_file=DATA_CONFIG["val_file"],
        root_dir=DATA_CONFIG["root_dir"],
        transform=val_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=TRAINING_CONFIG["num_workers"],
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        drop_last=True,
        shuffle=False,
        num_workers=TRAINING_CONFIG["num_workers"],
    )

    model_config = MODEL_CONFIGS[args.model].copy()
    model_config["pretrained"] = args.pretrained
    model = get_model(
        args.model, num_classes=DATA_CONFIG["num_classes"], **model_config
    )
    model = model.to(device)

    # Training configurations
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    # criterion = FocalLoss(alpha=0.75, gamma=2.0)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=TRAINING_CONFIG["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, mode="min", factor=0.1, patience=5
    )

    checkpoint_saver = CheckpointSaver(model, optimizer, save_dir=exp_dir)

    early_stopping = EarlyStopping(
        patience=TRAINING_CONFIG["patience"],
        min_delta=TRAINING_CONFIG["min_delta"],
        checkpoint_saver=checkpoint_saver,
    )

    trainer = SegmentationTrainer(model, optimizer, criterion, device)

    train_losses = []
    val_losses = []

    # Training
    for epoch in range(args.epochs):
        train_loss = trainer.train_epoch(train_loader, epoch)
        val_loss = trainer.validate(val_loader, epoch)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        # Save logs
        loss_path = os.path.join(exp_dir, f"{args.exp_name}_losses.pkl")
        with open(loss_path, "wb") as f:
            pickle.dump({"train": train_losses, "val": val_losses}, f)

        print(f"Epoch {epoch+1}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")

        if early_stopping(val_loss, epoch):
            break

    plot_path = os.path.join(
        OUTPUT_CONFIG["plots_dir"], args.exp_name, "loss_curves.png"
    )
    plot_loss_curves(train_losses, val_losses, save_path=plot_path)

    # Save experiment config
    config = {
        "model": args.model,
        "pretrained": args.pretrained,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "num_classes": DATA_CONFIG["num_classes"],
        "model_config": model_config,
    }

    with open(os.path.join(exp_dir, "config.pkl"), "wb") as f:
        pickle.dump(config, f)

    print(f"Training complete. Results saved in {exp_dir}")


if __name__ == "__main__":
    main()
