import torch
import torch.nn as nn

from models.attention_modules import CBAM
from models.resnet50 import ResNet50Decoder, ResNet50Encoder


class AttentionEarlyFusionHHA(nn.Module):
    def __init__(
        self,
        num_classes,
        dropout=0.3,
    ):
        super().__init__()

        self.encoder = ResNet50Encoder()
        self._adapt_input_channels()

        self.decoder = ResNet50Decoder(num_channels=num_classes, dropout=dropout)

        self.attention = nn.ModuleList([
            CBAM(in_channels=256, reduction_ratio=8),
            CBAM(in_channels=512, reduction_ratio=8),
            CBAM(in_channels=1024, reduction_ratio=8),
            CBAM(in_channels=2048, reduction_ratio=8)
        ])

    def forward(self, x):
        x = self.encoder(x)
        x = [self.attention[i](feat) for i, feat in enumerate(x)]
        x = self.decoder(x[-1], x[:-1])
        return x

    def get_optimizer_groups(self):
        first_layer = list(self.encoder.encoder.conv1.parameters())
        encoder = [p for name, p in self.encoder.encoder.named_parameters() if "conv1" not in name]
        decoder = list(self.decoder.parameters())

        return [
            {"params": first_layer, "lr": 5e-3},        
            {"params": encoder, "lr": 5e-4},
            {"params": decoder, "lr": 5e-3},
            {"params": self.attention.parameters(), "lr": 5e-3}
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
