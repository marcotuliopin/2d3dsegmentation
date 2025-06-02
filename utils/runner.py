import torch
from tqdm import tqdm


class Runner:
    def __init__(
        self,
        model,
        device,
        optimizer=None,
        criterion=None,
        rgb_only=False,
        use_hha=False,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.rgb_only = rgb_only
        self.use_hha = use_hha

    def train_epoch(self, loader):
        self.model.train()
        running_loss = 0.0

        for batch in tqdm(loader, desc=f"training epoch"):
            if self.rgb_only:
                images, masks = batch
            elif self.use_hha:
                images, masks, hha = batch
                images = torch.cat((images, hha), dim=1)
            else:
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
                elif self.use_hha:
                    images, masks, hha = batch
                    images = torch.cat((images, hha), dim=1)
                else:
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
                bincount = torch.bincount(idx.flatten(), minlength=k * k)
                conf_matrix += bincount.reshape(k, k)

        intersection = torch.diag(conf_matrix)
        union = conf_matrix.sum(1) + conf_matrix.sum(0) - intersection
        iou = intersection.float() / (union.float() + 1e-6)
        miou = iou.mean().item()

        loss = running_loss / len(loader)
        return loss, miou

    def test(self, loader):
        all_preds = []
        all_labels = []

        self.model.eval()

        with torch.no_grad():
            for batch in tqdm(loader):
                if self.rgb_only:
                    images, masks = batch
                elif self.use_hha:
                    images, masks, hha = batch
                    images = torch.cat((images, hha), dim=1)
                else:
                    images, masks, depth = batch
                    images = torch.cat((images, depth), dim=1)

                images = images.to(self.device)
                masks = masks.to(self.device)

                outputs = self.model(images)
                if isinstance(outputs, dict):
                    outputs = outputs["out"]

                preds = torch.argmax(outputs, dim=1)

                all_preds.append(preds)
                all_labels.append(masks)

        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        return all_preds, all_labels
