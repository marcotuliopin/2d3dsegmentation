import os
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import torch
import pickle
import argparse
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import DATA_CONFIG, OUTPUT_CONFIG, TRAINING_CONFIG
from models.segmentation import get_model
from utils.datasets import NYUDepthV2Dataset
from utils.transforms import get_validation_transforms
from utils.training import CheckpointSaver
from utils.visualization import plot_confusion_matrix, visualize_predictions
from utils.metrics import evaluate_batch


def parse_args():
    parser = argparse.ArgumentParser(description="Avaliação do modelo de segmentação")
    parser.add_argument(
        "-n",
        "--exp-name",
        type=str,
        required=True,
        help="Nome do experimento para avaliar",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=TRAINING_CONFIG["batch_size"],
        help="Tamanho do batch",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="latest",
        help='Nome do checkpoint específico (ou "latest")',
    )
    parser.add_argument(
        "--database",
        type=str,
        default="nyu_depth_v2",
        choices=["sun_rgbd", "nyu_depth_v2"],
        help="Dataset a ser usado",
    )
    return parser.parse_args()


def get_latest_checkpoint(checkpoint_dir):
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_dir}")

    sorted_checkpoints = sorted(checkpoints, key=lambda x: int(x.split("_")[-1].split(".")[0]))
    return os.path.join(checkpoint_dir, sorted_checkpoints[-1])


def test_model(model, data_loader, device, num_classes):
    all_preds = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for images, masks in tqdm(data_loader):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)["out"]
            preds = torch.argmax(outputs, dim=1)

            all_preds.append(preds)
            all_labels.append(masks)

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    return compute_segmentation_metrics(
        preds=all_preds,
        labels=all_labels,
        num_classes=num_classes,
        ignore_index=0
    )


def compute_segmentation_metrics(preds, labels, num_classes, ignore_index=0):
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

    # Pixel accuracy
    pixel_acc = intersection.sum() / np.maximum(conf_matrix.sum(), 1)

    # Precision, recall, F1 per class
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


def main():
    args = parse_args()

    print("CUDA Available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print(f"GPU Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"GPU Allocated Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"GPU Reserved Memory: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        print("CUDA Version:", torch.version.cuda)
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    exp_dir = os.path.join(OUTPUT_CONFIG["checkpoints_dir"], args.exp_name)
    with open(os.path.join(exp_dir, "config.pkl"), "rb") as f:
        config = pickle.load(f)

    model = get_model(config["model"], num_classes=config["num_classes"], **{
            k: v
            for k, v in config.get("model_config", {}).items()
            if k != "num_classes"
        },
    )
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    checkpoint_saver = CheckpointSaver(model, optimizer, exp_dir)

    if args.checkpoint == "latest":
        checkpoint_path = get_latest_checkpoint(exp_dir)
    else:
        checkpoint_path = os.path.join(exp_dir, args.checkpoint)
    checkpoint_saver.load(checkpoint_path)

    data_config = DATA_CONFIG[args.database]
    test_transform = get_validation_transforms(height=data_config["image_height"], width=data_config["image_width"])
    test_dataset = NYUDepthV2Dataset(data_config["test_file"], 
        transform=test_transform, 
        split_name="test"
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.get("batch_size", TRAINING_CONFIG["batch_size"]),
        drop_last=True,
        shuffle=False,
        num_workers=TRAINING_CONFIG["num_workers"]
    )

    results = test_model(model, test_loader, device, num_classes=config["num_classes"])
    results_dir = os.path.join(OUTPUT_CONFIG["results_dir"], args.exp_name)
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "test_results.pkl"), "wb") as f:
        pickle.dump(results, f)

    plots_dir = os.path.join(OUTPUT_CONFIG["plots_dir"], args.exp_name)
    os.makedirs(plots_dir, exist_ok=True)

    cm_path = os.path.join(plots_dir, "confusion_matrix.png")
    plot_confusion_matrix(results["confusion_matrix"], save_path=cm_path)

    vis_path = os.path.join(plots_dir, "predictions.png")
    visualize_predictions(model, test_loader, device, num_samples=4, save_path=vis_path)

    with open(os.path.join(results_dir, "test_results.txt"), "w") as f:
        f.write(f"Mean IoU: {results['mean_iou']:.4f}\n")
        f.write(f"Weighted IoU: {results['weighted_iou']:.4f}\n")
        f.write(f"Mean Dice: {results['mean_dice']:.4f}\n")
        f.write(f"Pixel Accuracy: {results['pixel_acc']:.4f}\n")
        f.write(f"F1 Score: {results['mean_f1']:.4f}\n")
        f.write("\nIoU por classe:\n")
        for i in range(config["num_classes"]):
            f.write(f"  Classe {i}: {results['class_iou'][i]:.4f}\n")


if __name__ == "__main__":
    main()
