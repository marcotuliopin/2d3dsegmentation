import torch
import torch.nn as nn
from torchvision import models

class ResNet50Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        # We are using the ResNet50 model from torchvision to be able to easily load the pretrained weights
        self.encoder = models.resnet50(weights=models.resnet.ResNet50_Weights.DEFAULT)
        self.out_channels = (64, 256, 512, 1024, 2048)

    def forward(self, x):
        features = []
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        features.append(x)

        x = self.encoder.maxpool(x)

        x = self.encoder.layer1(x)
        features.append(x)

        x = self.encoder.layer2(x)
        features.append(x)

        x = self.encoder.layer3(x)
        features.append(x)

        x = self.encoder.layer4(x)
        features.append(x)

        return features
