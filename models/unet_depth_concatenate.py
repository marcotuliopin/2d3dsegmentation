import torch.nn as nn
import segmentation_models_pytorch as smp


class UNetD1(nn.Module):
    def __init__(
        self,
        num_classes,
        encoder="resnet50",
        dropout=0.3,
        pretrained=True,
    ):
        super().__init__()

        weights = "imagenet" if pretrained else None

        self.unet = smp.Unet(
            encoder_name=encoder,
            encoder_weights=weights,
            in_channels=4, # RGB-D information
            classes=num_classes,
            activation=None,
        )

        # Add dropout in all decoder blocks
        for i in range(len(self.unet.decoder.blocks)):
            self.unet.decoder.blocks[i].dropout = nn.Dropout2d(p=dropout)

    def forward(self, x):
        return self.unet(x)

    def set_trainable(self, trainable=True):
        for param in self.unet.encoder.parameters():
            param.requires_grad = trainable


def get_unet_depth_concatenate(num_classes, dropout=0.3, pretrained=True, encoder="resnet50"):
    print(f"Using model: UNet with {encoder} encoder. Depth concatenation enabled.")
    return UNetD1(
        num_classes=num_classes,
        dropout=dropout,
        pretrained=pretrained,
        encoder=encoder,
    )
