import torch
import torch.nn as nn
import torch.nn.functional as F

# Frequency weights for different classes in the NYUv2 dataset
nyuv2_inv_freq = [0.11756749, 0.58930845, 3.86320268, 1.42978694, 0.61211152,
        0.21107389, 0.14174245, 0.16072167, 1.03913962, 0.87946776,
        0.68799929, 3.74469765, 0.08783193, 0.43534866]


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, reduction=None, ignore_index=255):
        super().__init__()
        self.alpha = torch.tensor(nyuv2_inv_freq)
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Tensor (N, C, H, W) - logits
            targets: Tensor (N, H, W) - ground truth labels
        """
        log_pt = F.log_softmax(inputs, dim=1)
        
        if self.ignore_index >= 0:
            valid_mask = (targets != self.ignore_index)
        else:
            valid_mask = torch.ones_like(targets, dtype=torch.bool)
        
        # Flatten tensors to simplify indexing
        log_pt = log_pt.permute(0, 2, 3, 1).contiguous().view(-1, inputs.size(1))
        targets_flat = targets.view(-1)
        valid_mask_flat = valid_mask.view(-1)
        
        # Filter out ignored indices
        log_pt = log_pt[valid_mask_flat]
        targets_flat = targets_flat[valid_mask_flat]
        
        if len(targets_flat) == 0:
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)
        
        log_pt = log_pt.gather(1, targets_flat.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()
        
        focal_weight = (1 - pt) ** self.gamma
        
        # Ajust weights based on alpha if provided
        if self.alpha.device != inputs.device:
            self.alpha = self.alpha.to(inputs.device)
            
        alpha_t = self.alpha[targets_flat]
        focal_weight = alpha_t * focal_weight
        
        focal_loss = -focal_weight * log_pt
        
        # Reduction transforms a batch of losses into a single scalar
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5, ignore_index=None):
        super().__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits, targets):
        num_classes = logits.shape[1]

        # Apply softmax to get class probabilities
        probs = torch.softmax(logits, dim=1)

        # Create mask for ignored indices if needed
        mask = None
        if self.ignore_index is not None:
            mask = (targets != self.ignore_index).float()

        # Initialize dice score
        dice_score = 0.0

        # Calculate dice for each class (avoiding one-hot encoding)
        for cls in range(num_classes):
            # Binary mask for current class
            target_cls = (targets == cls).float()
            prob_cls = probs[:, cls]

            # Apply ignore mask if needed
            if mask is not None:
                target_cls = target_cls * mask
                prob_cls = prob_cls * mask

            # Flatten tensors for simpler computation
            target_cls_flat = target_cls.reshape(-1)
            prob_cls_flat = prob_cls.reshape(-1)

            # Calculate intersection and union
            intersection = (prob_cls_flat * target_cls_flat).sum()
            union = prob_cls_flat.sum() + target_cls_flat.sum()

            # Calculate dice coefficient for this class
            class_dice = (2. * intersection + self.smooth) / (union + self.smooth)
            dice_score += class_dice

        # Average over all classes
        return 1.0 - (dice_score / num_classes)
