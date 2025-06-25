import torch
import torch.nn as nn

from models.resnet50 import ResNet50Encoder
from models.unet import UNetDecoder


class AttentionEarlyFusionHHA(nn.Module):
    def __init__(
        self,
        num_classes,
        dropout=0.3,
    ):
        super().__init__()

        self.encoder = ResNet50Encoder()
        self._adapt_input_channels()

        self.decoder = UNetDecoder(encoder_channels=self.encoder.out_channels, num_classes=num_classes)

        # self.attention = nn.ModuleList([
        #     CBAM(in_channels=1024, reduction_ratio=8),
        #     CBAM(in_channels=2048, reduction_ratio=8)
        # ])

    def forward(self, x):
        features = []
        x = self.encoder.encoder.conv1(x)
        x = self.encoder.encoder.bn1(x)
        x = self.encoder.encoder.relu(x)
        features.append(x)

        x = self.encoder.encoder.maxpool(x)

        x = self.encoder.encoder.layer1(x)
        features.append(x)

        x = self.encoder.encoder.layer2(x)
        features.append(x)

        x = self.encoder.encoder.layer3(x)
        x = self.attention[0](x)
        features.append(x)

        x = self.encoder.encoder.layer4(x)
        x = self.attention[1](x)
        features.append(x)

        output = self.decoder(features)
        return output

    def get_optimizer_groups(self):
        encoder = list(self.encoder.encoder.parameters())
        decoder = list(self.decoder.parameters())
        attention_parameters = list(self.attention.parameters())

        return [
            {"params": encoder, "lr": 1e-4},
            {"params": attention_parameters, "lr": 3e-4},
            {"params": decoder, "lr": 5e-4},
        ]

    def _adapt_input_channels(self):
        first_conv = self.encoder.encoder.conv1

        new_conv = nn.Conv2d(
            6, 
            first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=first_conv.bias is not None
        )

        with torch.no_grad():
            new_conv.weight.data[:, :3, :, :] = first_conv.weight.data
            new_conv.weight.data[:, 3:, :, :] = first_conv.weight.data
            if first_conv.bias is not None:
                new_conv.bias.data = first_conv.bias.data.clone()

        self.encoder.encoder.conv1 = new_conv
