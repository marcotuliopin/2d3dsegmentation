import torch
import torch.nn as nn
from torch.nn import functional as F

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
            {"params": self.fuse.parameters(), "lr": 5e-3}
        ]

    def _adapt_input_channels(self):
        first_conv = self.encoder.encoder.conv1  # For Resnet

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


class SAM(nn.Module):
    """Spatial Attention Module (SAM) for enhancing feature maps with spatial attention.
    """
    def __init__(self, bias=False):
        super(SAM, self).__init__()
        self.bias = bias
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3, dilation=1, bias=self.bias)

    def forward(self, x):
        max = torch.max(x,1)[0].unsqueeze(1)
        avg = torch.mean(x,1).unsqueeze(1)
        concat = torch.cat((max,avg), dim=1)
        output = self.conv(concat)
        output = F.sigmoid(output) * x 
        return output 


class CAM(nn.Module):
    """
    Channel Attention module (CAM) for cross-attention between RGB and HHA features.
    """
    def __init__(self, in_channels, reduction_ratio=8):
        super().__init__()
        self.reduced_dim = in_channels // reduction_ratio
        if self.reduced_dim < 1:
            self.reduced_dim = in_channels
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, self.reduced_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.reduced_dim, in_channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, q, k):
        B, C, _, _ = k.shape
        q_max = self.max_pool(q).view(B, C)
        q_avg = self.global_pool(q).view(B, C)
        lin_max = self.mlp(q_max).view(B, C, 1, 1)  
        lin_avg = self.mlp(q_avg).view(B, C, 1, 1)
        channel_weights = lin_max + lin_avg
        return k * self.sigmoid(channel_weights)


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM) for cross-attention between RGB and HHA features.
    """
    def __init__(self, in_channels, reduction_ratio=8):
        super().__init__()
        self.channel_attention = CAM(in_channels, reduction_ratio)
        self.spatial_attention = SAM()

    def forward(self, x):
        y = self.channel_attention(x, x)
        y = self.spatial_attention(y)
        return y