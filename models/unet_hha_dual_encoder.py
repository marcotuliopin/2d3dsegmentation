import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder


class UnetDualEncoderHHA(nn.Module):
    def __init__(self, num_classes, encoder="resnet50", dropout=0.3, pretrained=True):
        super().__init__()

        rgb_weights = "imagenet" if pretrained else None

        # Encoder RGB
        self.rgb_encoder = get_encoder(
            name=encoder,
            in_channels=3,
            weights=rgb_weights
        )

        # Encoder HHA
        self.hha_encoder = get_encoder(
            name=encoder,
            in_channels=3,
            weights=None
        )

        if pretrained:
            self._adapt_input_channels()
        
        # Batch normalization for each encoder output
        self.rgb_norms = nn.ModuleList([
            nn.BatchNorm2d(ch) for ch in self.rgb_encoder.out_channels
        ])
        self.hha_norms = nn.ModuleList([
            nn.BatchNorm2d(ch) for ch in self.hha_encoder.out_channels
        ])

        # Balance weights for the attention output. Adjust the balance between RGB and HHA features.
        self.balance_weights = nn.Parameter(torch.tensor(0.5))

        # Double of channels since we concatenate RGB and Depth features
        self.encoder_channels = [
            r + h for r, h in zip(
                self.rgb_encoder.out_channels,
                self.hha_encoder.out_channels
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
        hha = x[:, 3:, :, :]

        rgb_feats = self.rgb_encoder(rgb)
        hha_feats = self.hha_encoder(hha)

        # Fusion of features using concatenation
        rgb_feats_norm = [norm(feat) for feat, norm in zip(rgb_feats, self.rgb_norms)]
        hha_feats_norm = [norm(feat) for feat, norm in zip(hha_feats, self.hha_norms)]
        alpha = torch.sigmoid(self.balance_weights) # Adjust the balance between RGB and HHA features
        fused = [torch.cat([alpha * rgb_feat, (1 - alpha) * hha_feat], dim=1) 
                 for rgb_feat, hha_feat in zip(rgb_feats_norm, hha_feats_norm)]

        decoder_output = self.decoder(fused)
        dropout_output = self.dropout(decoder_output) 
        return self.segmentation_head(dropout_output)
    
    def get_optimizer_groups(self):
        rgb_encoder = [p for name, p in self.rgb_encoder.named_parameters() if "conv1" not in name]
        hha_encoder = [p for name, p in self.hha_encoder.named_parameters() if "conv1" not in name]

        decoder = list(self.decoder.parameters()) + list(self.segmentation_head.parameters())

        return [
            {"params": self.rgb_encoder.conv1.parameters(), "lr": 5e-4},
            {"params": self.hha_encoder.conv1.parameters(), "lr": 5e-3},
            {"params": rgb_encoder, "lr": 1e-4},
            {"params": hha_encoder, "lr": 1e-3},
            {"params": self.balance_weights, "lr": 1e-3},
            {"params": self.rgb_norms.parameters(), "lr": 1e-3},
            {"params": self.hha_norms.parameters(), "lr": 1e-3},
            {"params": decoder, "lr": 1e-2},
        ]
    
    def _adapt_input_channels(self):
        first_conv_rgb = self.rgb_encoder.conv1
        first_conv_hha = self.hha_encoder.conv1
        
        new_conv_hha = nn.Conv2d(
            3,  # HHA channel
            out_channels=first_conv_hha.out_channels,
            kernel_size=first_conv_hha.kernel_size,
            stride=first_conv_hha.stride,
            padding=first_conv_hha.padding,
            bias=first_conv_hha.bias is not None,
        )

        with torch.no_grad():
            # Initialize the HHA channel weights with the average of the RGB channels
            new_conv_hha.weight.data = first_conv_rgb.weight.data.clone()
            if first_conv_hha.bias is not None:
                new_conv_hha.bias.data = first_conv_hha.bias.data.clone()

        self.hha_encoder.conv1 = new_conv_hha


def get_unet_hha_dual_encoder(num_classes, dropout=0.3, pretrained=True, encoder="resnet50"):
    print(f"Using Unet with {encoder}. Using dual encoders for RGB and HHA inputs.")
    return UnetDualEncoderHHA(num_classes=num_classes, dropout=dropout, pretrained=pretrained, encoder=encoder)
