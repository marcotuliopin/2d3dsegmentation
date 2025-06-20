import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class UNetEarlyFusionD(nn.Module):
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
            in_channels=3,
            classes=num_classes,
            activation=None,
        )

        # Modify to accept 4 input channels (3 RGB + 1 D)
        if pretrained:
            self._adapt_input_channels()
        else:
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

    def get_optimizer_groups(self):
        first_layer = list(self.unet.encoder.conv1.parameters())
        encoder = [p for name, p in self.unet.encoder.named_parameters() if "conv1" not in name]
        decoder = list(self.unet.decoder.parameters()) + list(self.unet.segmentation_head.parameters())

        return [
            {"params": first_layer, "lr": 5e-3},        
            {"params": encoder, "lr": 1e-3},
            {"params": decoder, "lr": 1e-2},
        ]
    
    def _adapt_input_channels(self):
        first_conv = self.unet.encoder.conv1
        
        new_conv = nn.Conv2d(
            4,  # RGB + Depth
            out_channels=first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=first_conv.bias is not None,
        )

        with torch.no_grad():
            new_conv.weight.data[:, :3, :, :] = first_conv.weight.data
            # Initialize the depth channel weights with the average of the RGB channels
            new_conv.weight.data[:, 3, :, :] = new_conv.weight.data[:, :3, :, :].mean(dim=1)
            if first_conv.bias is not None:
                new_conv.bias.data = first_conv.bias.data.clone()
        
        self.unet.encoder.conv1 = new_conv


def get_unet_early_fusion_d(num_classes, dropout=0.3, pretrained=True, encoder="resnet50"):
    print(f"Using model: UNet with {encoder} encoder. Depth concatenation enabled.")
    return UNetEarlyFusionD(
        num_classes=num_classes,
        dropout=dropout,
        pretrained=pretrained,
        encoder=encoder,
    )
