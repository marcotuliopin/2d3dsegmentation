import torch.nn as nn
import segmentation_models_pytorch as smp


class UNet(nn.Module):
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
            in_channels=3, # RGB only
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
    
    def get_optimizer_groups(self):
        return {
            "first_layer": list(self.unet.encoder.conv1.parameters()),
            "encoder": [p for name, p in self.unet.encoder.named_parameters()
                        if "conv1" not in name],
            "decoder": list(self.unet.decoder.parameters())
            + list(self.unet.segmentation_head.parameters()),
        }


def get_unet(num_classes, dropout=0.3, pretrained=True, encoder="resnet50"):
    print(f"Using model: UNet with {encoder} encoder")
    return UNet(
        num_classes=num_classes,
        dropout=dropout,
        pretrained=pretrained,
        encoder=encoder,
    )
