import os
import torch
from tqdm import tqdm


class CheckpointSaver:
    def __init__(self, model, save_dir, optimizer=None):
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
        if self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        print(f"Checkpoint loaded from {checkpoint_path}, starting from epoch {start_epoch}")
        return start_epoch


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.counter = 0
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print("Early stopping triggered")
                return True
        return False


class Trainer:
    def __init__(self, model, optimizer, criterion, device, scheduler=None, rgb_only=False):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.rgb_only = rgb_only

    def train_epoch(self, loader):
        self.model.train()
        running_loss = 0.0
        
        for batch in tqdm(loader, desc=f"training epoch"):
            if self.rgb_only:
                images, masks = batch
            else:
                # Add depth as fourth channel to image
                images, masks, depth = batch
                images = torch.cat((images, depth), dim=1)

            images = images.to(self.device)
            masks = masks.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            if isinstance(outputs, dict):
                outputs = outputs["out"]

            loss = self.criterion(outputs, masks)
            running_loss += loss.item()

            loss.backward()

            self.optimizer.step()
            self.scheduler.step()
        
        loss = running_loss / len(loader)
        return loss

    def validate(self, loader):
        self.model.eval()
        running_loss = 0.0

        conf_matrix = torch.zeros(14, 14, device=self.device)
        
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"validation epoch"):
                if self.rgb_only:
                    images, masks = batch
                else:
                    # Add depth as fourth channel to image
                    images, masks, depth = batch
                    images = torch.cat((images, depth), dim=1)

                images = images.to(self.device)
                masks = masks.to(self.device)

                outputs = self.model(images)
                if isinstance(outputs, dict):
                    outputs = outputs["out"]

                loss = self.criterion(outputs, masks)
                running_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                k = conf_matrix.size(0)
                idx = masks * k + preds
                bincount = torch.bincount(idx.flatten(), minlength=k*k)
                conf_matrix += bincount.reshape(k, k)
        
        intersection = torch.diag(conf_matrix)
        union = conf_matrix.sum(1) + conf_matrix.sum(0) - intersection
        iou = intersection.float() / (union.float() + 1e-6)
        miou = iou.mean().item()

        loss = running_loss / len(loader)
        return loss, miou
