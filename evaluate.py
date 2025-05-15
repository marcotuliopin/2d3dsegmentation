import os
import torch
import pickle
import argparse
import numpy as np
from torch.utils.data import DataLoader

from config import DATA_CONFIG, OUTPUT_CONFIG, TRAINING_CONFIG
from models.segmentation import get_model
from utils.datasets import SunRGBDDataset
from utils.transforms import get_validation_transforms
from utils.training import CheckpointSaver
from utils.visualization import plot_confusion_matrix, visualize_predictions
from utils.metrics import evaluate_batch


def parse_args():
    parser = argparse.ArgumentParser(description='Avaliação do modelo de segmentação')
    parser.add_argument('n', '--exp-name', type=str, required=True,
                        help='Nome do experimento para avaliar')
    parser.add_argument('--checkpoint', type=str, default='latest',
                        help='Nome do checkpoint específico (ou "latest")')
    return parser.parse_args()


def get_latest_checkpoint(checkpoint_dir):
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    if not checkpoints:
        raise FileNotFoundError(f"Nenhum checkpoint encontrado em {checkpoint_dir}")
    
    sorted_checkpoints = sorted(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    return os.path.join(checkpoint_dir, sorted_checkpoints[-1])


def test_model(model, data_loader, device, num_classes=14):
    model.eval()
    confusion_matrix = np.zeros((num_classes, num_classes))
    
    total_iou = 0.0
    total_dice = 0.0
    total_pixel_acc = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for images, masks in data_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)['out']
            preds = torch.argmax(outputs, dim=1)
            
            preds_np = preds.cpu().numpy()
            masks_np = masks.cpu().numpy()
            
            batch_cm, batch_metrics = evaluate_batch(preds_np, masks_np, num_classes)
            confusion_matrix += batch_cm
            
            total_iou += sum(batch_metrics['iou'])
            total_dice += sum(batch_metrics['dice'])
            total_pixel_acc += sum(batch_metrics['pixel_accuracy'])
            total_samples += len(batch_metrics['iou'])
    
    class_iou = np.diag(confusion_matrix) / (np.sum(confusion_matrix, axis=1) + 
                                             np.sum(confusion_matrix, axis=0) - 
                                             np.diag(confusion_matrix) + 1e-7)
    
    mean_iou = np.mean(class_iou)
    mean_pixel_acc = total_pixel_acc / total_samples
    mean_dice = total_dice / total_samples
    
    print(f"Métricas de teste:")
    print(f"  Mean IoU: {mean_iou:.4f}")
    print(f"  Mean Dice Coefficient: {mean_dice:.4f}")
    print(f"  Pixel Accuracy: {mean_pixel_acc:.4f}")
    
    print("\nIoU por classe:")
    for i in range(num_classes):
        print(f"  Classe {i}: {class_iou[i]:.4f}")
    
    return {
        'mean_iou': mean_iou,
        'mean_dice': mean_dice,
        'pixel_acc': mean_pixel_acc,
        'class_iou': class_iou,
        'confusion_matrix': confusion_matrix
    }


def main():
    args = parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    
    exp_dir = os.path.join(OUTPUT_CONFIG['checkpoints_dir'], args.exp_name)
    
    with open(os.path.join(exp_dir, 'config.pkl'), 'rb') as f:
        config = pickle.load(f)
    
    model = get_model(
        config['model'],
        num_classes=config['num_classes'],
        **{k: v for k, v in config.get('model_config', {}).items() if k != 'num_classes'}
    )
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    checkpoint_saver = CheckpointSaver(model, optimizer, exp_dir)
    
    if args.checkpoint == 'latest':
        checkpoint_path = get_latest_checkpoint(exp_dir)
    else:
        checkpoint_path = os.path.join(exp_dir, args.checkpoint)
    
    checkpoint_saver.load(checkpoint_path)
    
    test_transform = get_validation_transforms(
        height=DATA_CONFIG['image_height'],
        width=DATA_CONFIG['image_width']
    )
    
    test_dataset = SunRGBDDataset(
        path_file=DATA_CONFIG['test_file'],
        root_dir=DATA_CONFIG['root_dir'],
        transform=test_transform
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.get('batch_size', TRAINING_CONFIG['batch_size']),
        shuffle=False,
        num_workers=TRAINING_CONFIG['num_workers']
    )
    
    results = test_model(model, test_loader, device, num_classes=config['num_classes'])
    
    results_dir = os.path.join(OUTPUT_CONFIG['results_dir'], args.exp_name)
    os.makedirs(results_dir, exist_ok=True)
    
    with open(os.path.join(results_dir, 'test_results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    plots_dir = os.path.join(OUTPUT_CONFIG['plots_dir'], args.exp_name)
    os.makedirs(plots_dir, exist_ok=True)
    
    cm_path = os.path.join(plots_dir, 'confusion_matrix.png')
    plot_confusion_matrix(results['confusion_matrix'], save_path=cm_path)
    
    vis_path = os.path.join(plots_dir, 'predictions.png')
    visualize_predictions(model, test_loader, device, num_samples=4, save_path=vis_path)
    
    print(f"Resultados finais do teste:")
    print(f"IoU médio: {results['mean_iou']:.4f}")
    print(f"Dice médio: {results['mean_dice']:.4f}")
    print(f"Acurácia de pixel: {results['pixel_acc']:.4f}")

if __name__ == "__main__":
    main()