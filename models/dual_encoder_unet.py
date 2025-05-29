import torch
import torch.nn as nn
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder

class DualEncoderUNet(nn.Module):
    def __init__(self, num_classes, encoder="resnet50", dropout=0.3, pretrained=True):
        super().__init__()

        # Encoder RGB
        self.rgb_encoder = get_encoder(
            name=encoder,
            in_channels=3,
            weights="imagenet" if pretrained else None
        )

        # Encoder Depth
        self.d_encoder = get_encoder(
            name=encoder,
            in_channels=1,
            weights=None
        )

        # Double of channels since we concatenate RGB and Depth features
        self.encoder_channels = [
            r + d for r, d in zip(
                self.rgb_encoder.out_channels,
                self.d_encoder.out_channels
            )
        ]
        print("Encoder channels:", self.encoder_channels)

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
        print("Input shapes:")
        print(rgb.shape, depth.shape)

        rgb_feats = self.rgb_encoder(rgb)
        d_feats = self.d_encoder(depth)

        print("RGB Features:", rgb_feats.shape)
        print("Depth Features:", d_feats.shape)

        # Fusion of features using concatenation
        feats = [torch.cat([r, d], dim=1) for r, d in zip(rgb_feats, d_feats)]

        decoder_output = self.decoder(feats)
        decoder_output = self.dropout(decoder_output)
        out = self.segmentation_head(decoder_output)
        return out


def get_dual_encoder_unet(num_classes, dropout=0.3, pretrained=True, encoder="resnet50", freeze_backbone=False, in_channels=3):
    print(f"Using DualEncoder UNet with encoder {encoder}")
    return DualEncoderUNet(num_classes=num_classes, dropout=dropout, pretrained=pretrained, encoder=encoder)
