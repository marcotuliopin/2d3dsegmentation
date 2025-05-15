import numpy as np


def calculate_iou(pred, target):
    intersection = np.logical_and(pred, target)
    union = np.logical_or(pred, target)
    iou_score = np.sum(intersection) / (np.sum(union) + 1e-7)
    return iou_score


def calculate_dice(pred, target):
    intersection = np.sum(np.logical_and(pred, target))
    dice_score = (2.0 * intersection) / (np.sum(pred) + np.sum(target) + 1e-7)
    return dice_score


def calculate_pixel_accuracy(pred, target):
    return np.mean(pred == target)


def evaluate_batch(preds, masks, num_classes):
    batch_size = preds.shape[0]
    confusion_matrix = np.zeros((num_classes, num_classes))
    metrics = {
        'iou': [], 
        'dice': [], 
        'pixel_accuracy': []
    }
    
    for i in range(batch_size):
        pred_flat = preds[i].flatten()
        mask_flat = masks[i].flatten()
        
        for t, p in zip(mask_flat, pred_flat):
            confusion_matrix[t, p] += 1
            
        metrics['iou'].append(calculate_iou(preds[i], masks[i]))
        metrics['dice'].append(calculate_dice(preds[i], masks[i]))
        metrics['pixel_accuracy'].append(calculate_pixel_accuracy(preds[i], masks[i]))
        
    return confusion_matrix, metrics