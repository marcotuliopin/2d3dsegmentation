import torch
import torch.nn as nn

from models.attention_modules import CAM
from models.resnet50 import ResNet50Decoder, ResNet50Encoder


class AttentionMidFusion(nn.Module):
    def __init__(self, num_classes, dropout=0.3, d_channels=1):
        super().__init__()

        self.rgb_encoder = ResNet50Encoder()
        self.d_encoder = ResNet50Encoder(dropout=dropout)
        self._adapt_input_channels(d_channels)
        
        self.rgb_norms = nn.ModuleList([nn.BatchNorm2d(ch) for ch in self.rgb_encoder.out_channels])
        self.d_norms = nn.ModuleList([nn.BatchNorm2d(ch) for ch in self.d_encoder.out_channels])

        self.rgb_modulator = nn.ModuleList([
            CAM(in_channels=64, reduction_ratio=8),
            CAM(in_channels=256, reduction_ratio=8),
            CAM(in_channels=512, reduction_ratio=8),
            CAM(in_channels=1024, reduction_ratio=8)
        ])
        self.d_modulator = nn.ModuleList([
            CAM(in_channels=64, reduction_ratio=8),
            CAM(in_channels=256, reduction_ratio=8),
            CAM(in_channels=512, reduction_ratio=8),
            CAM(in_channels=1024, reduction_ratio=8)
        ])

        self.decoder = ResNet50Decoder(num_channels=num_classes, dropout=dropout)

    def forward(self, x):
        rgb = x[:, :3, :, :]
        d = x[:, 3:, :, :]

        rgb_feats = []
        d_feats = []

        rgb = self.rgb_encoder.encoder.conv1(rgb)
        rgb = self.rgb_encoder.encoder.bn1(rgb)
        rgb = self.rgb_encoder.encoder.relu(rgb)
        rgb = self.rgb_encoder.encoder.maxpool(rgb)
        
        d = self.d_encoder.encoder.conv1(d)
        d = self.d_encoder.encoder.bn1(d)
        d = self.d_encoder.encoder.relu(d)
        d = self.d_encoder.encoder.maxpool(d)

        rgb = self.rgb_modulator[0](rgb, d)
        d = self.d_modulator[0](d, rgb)

        rgb = self.rgb_encoder.encoder.layer1(rgb)
        d = self.d_encoder.encoder.layer1(d)
        rgb_feats.append(rgb)
        d_feats.append(d)

        rgb = self.rgb_modulator[1](rgb, d)
        d = self.d_modulator[1](d, rgb)

        rgb = self.rgb_encoder.encoder.layer2(rgb)
        d = self.d_encoder.encoder.layer2(d)
        rgb_feats.append(rgb)
        d_feats.append(d)

        rgb = self.rgb_modulator[2](rgb, d)
        d = self.d_modulator[2](d, rgb)

        rgb = self.rgb_encoder.encoder.layer3(rgb)
        d = self.d_encoder.encoder.layer3(d)
        rgb_feats.append(rgb)
        d_feats.append(d)

        rgb = self.rgb_modulator[3](rgb, d)
        d = self.d_modulator[3](d, rgb)

        rgb = self.rgb_encoder.encoder.layer4(rgb)
        d = self.d_encoder.encoder.layer4(d)
        rgb_feats.append(rgb)
        d_feats.append(d)

        rgb_feats_norm = [norm(feat) for feat, norm in zip(rgb_feats, self.rgb_norms)]
        d_feats_norm = [norm(feat) for feat, norm in zip(d_feats, self.d_norms)]
        x = [r + h for r, h in zip(rgb_feats_norm, d_feats_norm)]

        x = self.decoder(x[-1], x[:-1])

        return x
    
    def get_optimizer_groups(self):
        rgb_encoder = [p for name, p in self.rgb_encoder.encoder.named_parameters() if "conv1" not in name]
        d_encoder = [p for name, p in self.d_encoder.encoder.named_parameters() if "conv1" not in name]

        decoder = list(self.decoder.parameters())

        return [
            {"params": self.rgb_encoder.encoder.conv1.parameters(), "lr": 5e-4},
            {"params": self.d_encoder.encoder.conv1.parameters(), "lr": 5e-3},
            {"params": rgb_encoder, "lr": 1e-4},
            {"params": d_encoder, "lr": 1e-3},
            {"params": self.rgb_norms.parameters(), "lr": 1e-3},
            {"params": self.d_norms.parameters(), "lr": 1e-3},
            {"params": decoder, "lr": 1e-2},
        ]
    
    def _adapt_input_channels(self, d_channels):
        first_conv_rgb = self.rgb_encoder.encoder.conv1
        first_conv_d = self.d_encoder.encoder.conv1
        
        new_conv_d = nn.Conv2d(
            d_channels,
            out_channels=first_conv_d.out_channels,
            kernel_size=first_conv_d.kernel_size,
            stride=first_conv_d.stride,
            padding=first_conv_d.padding,
            bias=first_conv_d.bias is not None,
        )

        with torch.no_grad():
            if d_channels == 1:
                # Average the RGB channels to initialize the depth channel
                new_conv_d.weight.data = first_conv_rgb.weight.data.mean(dim=1, keepdim=True)
            else:
                new_conv_d.weight.data = first_conv_rgb.weight.data.clone()
            if first_conv_d.bias is not None:
                new_conv_d.bias.data = first_conv_d.bias.data.clone()

        self.d_encoder.encoder.conv1 = new_conv_d
