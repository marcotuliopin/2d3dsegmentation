import os
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import torch
import pickle
import argparse
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from utils.datasets import NYUDepthV2Dataset
from utils.model import get_model
from utils.transforms import get_validation_transforms
from utils.training import CheckpointSaver
from utils.visualization import plot_confusion_matrix, visualize_predictions


def parse_args(config):
    parser = argparse.ArgumentParser(description="Model Evaluation")
    parser.add_argument(
        "-n",
        "--experiment_name",
        type=str,
        required=True,
        help="Experiment name",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=config["train"]["batch_size"],
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="nyu_depth_v2",
        choices=[key for key in config["data"].keys()],
        help="Dataset to be used",
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

    # Make sure the experiment exists
    exp_dir = os.path.join(config["output"]["directories"]["checkpoints"], args.experiment_name)
    with open(os.path.join(exp_dir, "config.pkl"), "rb") as f:
        exp_config = pickle.load(f)

    # We need to load the model with the same configuration used for training
    in_channels = 4 if exp_config["use_depth"] else 3
    model = get_model(exp_config["model"], 
        num_classes=exp_config["num_classes"], 
        in_channels=in_channels, **{
            k: v
            for k, v in exp_config.get("model_config", {}).items()
            if k != "num_classes"
        },
    )
    model = model.to(device)

    checkpoint_saver = CheckpointSaver(model, exp_dir)
    checkpoint_path = get_latest_checkpoint(exp_dir)
    checkpoint_saver.load(checkpoint_path)

    # ------ Data  Loading ------
    data_config = config["data"][args.data]

    test_dataset = NYUDepthV2Dataset(
        data_config["paths"]["test_file"], 
        mode="test",
        shape=config["data"]["shape"],
        use_depth=exp_config["use_depth"]
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        drop_last=True,
        shuffle=False,
        num_workers=config["train"]["num_workers"]
    )

    # ------ Testing ------
    results = test_model(model, test_loader, device, exp_config["num_classes"], data_config["unlabeled_id"])
    results_dir = os.path.join(config["output"]["directories"]["results"], args.experiment_name)

    # Make sure the output directories exist
    os.makedirs(results_dir, exist_ok=True)

    plots_dir = os.path.join(config["output"]["directories"]["plots"], args.experiment_name)
    os.makedirs(plots_dir, exist_ok=True)

    cm_path = os.path.join(plots_dir, "confusion_matrix.png")
    plot_confusion_matrix(results["confusion_matrix"], save_path=cm_path)

    with open(os.path.join(results_dir, "test_results.txt"), "w") as f:
        f.write(f"Mean IoU: {results['mean_iou']:.4f}\n")
        f.write(f"Weighted IoU: {results['weighted_iou']:.4f}\n")
        f.write(f"Mean Dice: {results['mean_dice']:.4f}\n")
        f.write(f"Pixel Accuracy: {results['pixel_acc']:.4f}\n")
        f.write(f"F1 Score: {results['mean_f1']:.4f}\n")
        f.write("\nIoU por classe:\n")
        for i in range(exp_config["num_classes"]):
            f.write(f"  Classe {i}: {results['class_iou'][i]:.4f}\n")

    vis_path = os.path.join(plots_dir, "predictions.png")
    visualize_predictions(model, test_loader, device, num_samples=4, save_path=vis_path)

def test_model(model, data_loader, device, num_classes, unlabeled_id):
    all_preds = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for images, masks in tqdm(data_loader):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            if isinstance(outputs, dict):
                outputs = outputs["out"]

            preds = torch.argmax(outputs, dim=1)

            all_preds.append(preds)
            all_labels.append(masks)

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    return compute_segmentation_metrics(
        preds=all_preds,
        labels=all_labels,
        num_classes=num_classes,
        ignore_index=unlabeled_id
    )


def compute_segmentation_metrics(preds, labels, num_classes, ignore_index=255):
    preds = preds.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()

    preds = preds.flatten()
    labels = labels.flatten()

    # Remove ignored index
    valid = labels != ignore_index
    preds = preds[valid]
    labels = labels[valid]

    conf_matrix = confusion_matrix(labels, preds, labels=list(range(num_classes)))
    intersection = np.diag(conf_matrix)
    ground_truth_set = conf_matrix.sum(axis=1)
    predicted_set = conf_matrix.sum(axis=0)
    union = ground_truth_set + predicted_set - intersection

    class_iou = intersection / np.maximum(union, 1)
    mean_iou = np.nanmean(class_iou)
    weighted_iou = np.sum(class_iou * ground_truth_set) / np.sum(ground_truth_set)

    # Dice = 2 * |A ∩ B| / (|A| + |B|) = 2*TP / (2*TP + FP + FN)
    class_dice = (2 * intersection) / np.maximum((ground_truth_set + predicted_set), 1)
    mean_dice = np.nanmean(class_dice)

    pixel_acc = intersection.sum() / np.maximum(conf_matrix.sum(), 1)

    precision = precision_score(labels, preds, average=None, labels=range(num_classes), zero_division=0)
    recall = recall_score(labels, preds, average=None, labels=range(num_classes), zero_division=0)
    f1_per_class = f1_score(labels, preds, average=None, labels=range(num_classes), zero_division=0)
    mean_f1 = np.nanmean(f1_per_class)

    print(f"Test metrics:")
    print(f"  Mean IoU: {mean_iou:.4f}")
    print(f"  Mean Dice Coefficient: {mean_dice:.4f}")
    print(f"  Pixel Accuracy: {pixel_acc:.4f}")
    print(f"  Weighted IoU: {weighted_iou:.4f}")

    print("\nIoU por classe:")
    for i in range(num_classes):
        print(f"  Classe {i}: {class_iou[i]:.4f}")

    return {
        "mean_iou": mean_iou,
        "weighted_iou": weighted_iou,
        "mean_dice": mean_dice,
        "pixel_acc": pixel_acc,
        "class_iou": class_iou,
        "confusion_matrix": conf_matrix,
        "precision": precision,
        "recall": recall,
        "f1_per_class": f1_per_class,
        "mean_f1": mean_f1,
    }


def get_latest_checkpoint(checkpoint_dir):
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_dir}")
    return os.path.join(checkpoint_dir, checkpoints[-1])


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


if __name__ == "__main__":
    config = read_config()
    args = parse_args(config)
    main(args, config)
