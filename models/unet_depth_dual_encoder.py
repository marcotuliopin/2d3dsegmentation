import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder


class UnetDualEncoderD(nn.Module):
    def __init__(self, num_classes, encoder="resnet50", dropout=0.3, pretrained=True):
        super().__init__()

        rgb_weights = "imagenet" if pretrained else None

        # Encoder RGB
        self.rgb_encoder = get_encoder(
            name=encoder,
            in_channels=3,
            weights=rgb_weights
        )

        # Encoder Depth
        self.d_encoder = get_encoder(
            name=encoder,
            in_channels=1,
            weights=None
        )

        if pretrained:
            self._adapt_input_channels()

        # Batch normalization for each encoder output
        self.rgb_norms = nn.ModuleList([
            nn.BatchNorm2d(ch) for ch in self.rgb_encoder.out_channels
        ])
        self.d_norms = nn.ModuleList([
            nn.BatchNorm2d(ch) for ch in self.d_encoder.out_channels
        ])

        # Double of channels since we concatenate RGB and Depth features
        self.encoder_channels = [
            r + d for r, d in zip(
                self.rgb_encoder.out_channels,
                self.d_encoder.out_channels
            )
        ]

        self.decoder = UnetDecoder(
            encoder_channels=self.encoder_channels, # Skip connections
            decoder_channels=(256, 128, 64, 32, 16),
            n_blocks=5,
        )

        self.segmentation_head = nn.Conv2d(16, num_classes, kernel_size=1)
        self.dropout = nn.Dropout2d(p=dropout)

    def forward(self, x):
        rgb = x[:, :3, :, :]
        depth = x[:, 3:, :, :]

        rgb_feats = self.rgb_encoder(rgb)
        d_feats = self.d_encoder(depth)

        # Fusion of features using concatenation
        rgb_feats_norm = [norm(feat) for feat, norm in zip(rgb_feats, self.rgb_norms)]
        d_feats_norm = [norm(feat) for feat, norm in zip(d_feats, self.d_norms)]
        feats = [torch.cat([r, d], dim=1) for r, d in zip(rgb_feats_norm, d_feats_norm)]

        decoder_output = self.decoder(feats)
        dropout_output = self.dropout(decoder_output) 
        return self.segmentation_head(dropout_output)
    
    def get_optimizer_groups(self):
        first_layer_rgb = list(self.rgb_encoder.conv1.parameters())
        first_layer_depth = list(self.d_encoder.conv1.parameters())

        rgb_encoder = [p for name, p in self.rgb_encoder.named_parameters() if "conv1" not in name]
        depth_encoder = [p for name, p in self.d_encoder.named_parameters() if "conv1" not in name]

        decoder = list(self.decoder.parameters()) + list(self.segmentation_head.parameters())

        return [
            {"params": first_layer_rgb, "lr": 5e-4},
            {"params": first_layer_depth, "lr": 5e-3},
            {"params": rgb_encoder, "lr": 1e-4},
            {"params": depth_encoder, "lr": 1e-3},
            {"params": decoder, "lr": 1e-2},
        ]
    
    def _adapt_input_channels(self):
        first_conv_rgb = self.rgb_encoder.conv1
        first_conv_depth = self.d_encoder.conv1
        
        new_conv_depth = nn.Conv2d(
            1,  # Depth channel
            out_channels=first_conv_depth.out_channels,
            kernel_size=first_conv_depth.kernel_size,
            stride=first_conv_depth.stride,
            padding=first_conv_depth.padding,
            bias=first_conv_depth.bias is not None,
        )

        with torch.no_grad():
            # Initialize the depth channel weights with the average of the RGB channels
            new_conv_depth.weight.data = first_conv_rgb.weight.data.mean(dim=1, keepdim=True)
            if first_conv_depth.bias is not None:
                new_conv_depth.bias.data = first_conv_depth.bias.data.clone()

        self.d_encoder.conv1 = new_conv_depth


def get_unet_depth_dual_encoder(num_classes, dropout=0.3, pretrained=True, encoder="resnet50"):
    print(f"Using Unet with {encoder}. Using dual encoders for RGB and Depth inputs.")
    return UnetDualEncoderD(num_classes=num_classes, dropout=dropout, pretrained=pretrained, encoder=encoder)
