import torch.nn as nn
import segmentation_models_pytorch as smp

class UNet(nn.Module):
    def __init__(self, num_classes, encoder="resnet50", dropout=0.2, in_channels=3, freeze_backbone=False, pretrained=True):
        super().__init__()
        self.unet = smp.Unet(
            encoder_name=encoder,
            encoder_weights="imagenet" if pretrained else None,
            in_channels=in_channels,
            classes=num_classes,
            activation=None
        )

        # Add dropout in all decoder blocks
        for i in range(len(self.unet.decoder.blocks)):
            self.unet.decoder.blocks[i].dropout = nn.Dropout2d(p=dropout)

    def forward(self, x):
        return self.unet(x)


def get_unet_resnet50(num_classes, dropout=0.2, in_channels=3, freeze_backbone=False, pretrained=True, encoder="resnet50"):
    print("Using model: UNet ResNet50")
    return UNet(num_classes=num_classes, dropout=dropout, in_channels=in_channels, freeze_backbone=freeze_backbone, pretrained=pretrained, encoder=encoder)