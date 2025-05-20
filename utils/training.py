import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class CheckpointSaver:
    def __init__(self, model, optimizer, save_dir):
        self.model = model
        self.optimizer = optimizer
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def save(self, epoch, verbose=False):
        checkpoint_path = os.path.join(self.save_dir, f"checkpoint.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, checkpoint_path)
        if verbose:
            print(f"Checkpoint saved at {checkpoint_path}")

    def load(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        print(f"Checkpoint loaded from {checkpoint_path}, starting from epoch {start_epoch}")
        return start_epoch


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0, checkpoint_saver=None):
        self.counter = 0
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.checkpoint_saver = checkpoint_saver

    def __call__(self, val_loss, epoch):
        if val_loss < self.best_loss - self.min_delta:
            self.checkpoint_saver.save(epoch)
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print("Early stopping triggered")
                return True
        return False


class SegmentationTrainer:
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train_epoch(self, data_loader, epoch):
        self.model.train()
        running_loss = 0.0
        
        for images, masks in tqdm(data_loader, desc=f"Training Epoch {epoch}"):
            images = images.to(self.device)
            masks = masks.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)['out']
            loss = self.criterion(outputs, masks)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            
        return running_loss / len(data_loader)

    def validate(self, data_loader, epoch):
        self.model.eval()
        running_loss = 0.0
        
        with torch.no_grad():
            for images, masks in tqdm(data_loader, desc=f"Validation Epoch {epoch}"):
                images = images.to(self.device)
                masks = masks.to(self.device)

                outputs = self.model(images)['out']
                loss = self.criterion(outputs, masks)

                running_loss += loss.item()
                
        return running_loss / len(data_loader)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, ignore_index=None):
        super().__init__()
        self.alpha = alpha  # Balances positive/negative
        self.gamma = gamma  # Focuses on hard examples
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        if self.ignore_index is not None:
            ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)
        else:
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # Softmax probability of true class
        focal_loss = (self.alpha * (1-pt)**self.gamma * ce_loss)
        return focal_loss.mean()