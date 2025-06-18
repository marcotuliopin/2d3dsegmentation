import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder


class UnetAttentionHHA(nn.Module):
    def __init__(self, num_classes, encoder="resnet50", dropout=0.3, pretrained=True):
        super().__init__()

        rgb_weights = "imagenet" if pretrained else None

        self.rgb_encoder = get_encoder(name=encoder, in_channels=3, weights=rgb_weights, dropout=dropout)
        self.hha_encoder = get_encoder(name=encoder, in_channels=3, weights=None, dropout=dropout)

        if pretrained:
            self._adapt_input_channels()
        
        # Batch normalization for each encoder output
        self.rgb_norms = nn.ModuleList([nn.BatchNorm2d(ch) for ch in self.rgb_encoder.out_channels])
        self.hha_norms = nn.ModuleList([nn.BatchNorm2d(ch) for ch in self.hha_encoder.out_channels])

        self.fuse = nn.ModuleList([FRM(in_channels=ch, reduction_ratio=8, dropout=dropout) for ch in self.rgb_encoder.out_channels])

        # Double of channels since we concatenate RGB and Depth features
        self.encoder_channels = [
            r + h for r, h in zip(
                self.rgb_encoder.out_channels,
                self.hha_encoder.out_channels
            )
        ]
        self.decoder = UnetDecoder(
            encoder_channels=self.encoder_channels, # Skip connections
            decoder_channels=(512, 256, 128, 64, 32),
            n_blocks=5,
        )

        self.segmentation_head = nn.Conv2d(32, num_classes, kernel_size=1)
        self.dropout = nn.Dropout2d(p=dropout)

    def forward(self, x):
        rgb = x[:, :3, :, :]
        hha = x[:, 3:, :, :]

        rgb_feats = self.rgb_encoder(rgb)
        hha_feats = self.hha_encoder(hha)

        # Fusion of features using concatenation
        rgb_feats_norm = [norm(feat) for feat, norm in zip(rgb_feats, self.rgb_norms)]
        hha_feats_norm = [norm(feat) for feat, norm in zip(hha_feats, self.hha_norms)]
        fused = [self.fuse[i](rgb_feats_norm[i], hha_feats_norm[i]) for i in range(len(rgb_feats_norm))]

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
            {"params": self.rgb_norms.parameters(), "lr": 1e-3},
            {"params": self.hha_norms.parameters(), "lr": 1e-3},
            {"params": self.fuse.parameters(), "lr": 1e-3},
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


class FRM(nn.Module):
    """
    Fusion Refinement Module (FRM) for cross-attention between RGB and HHA features.
    """
    def __init__(self, in_channels, reduction_ratio=8, dropout=0.3):
        super().__init__()
        self.rgb_att = CBAM(in_channels, reduction_ratio)
        self.hha_att = CBAM(in_channels, reduction_ratio)
        self.rgb_hha_att = CAM(in_channels, reduction_ratio)
        self.hha_rgb_att = CAM(in_channels, reduction_ratio)
        self.dropout = nn.Dropout2d(p=dropout)

    def forward(self, rgb_feat, hha_feat):
        rgb_ref = self.rgb_att(rgb_feat)
        hha_ref = self.hha_att(hha_feat)
        return torch.cat([rgb_ref, hha_ref], dim=1)
        rgb_to_hha = self.rgb_hha_att(rgb_ref, hha_ref)
        hha_to_rgb = self.hha_rgb_att(hha_ref, rgb_ref)
        output = torch.cat([rgb_to_hha, hha_to_rgb], dim=1)
        return self.dropout(output)


def get_unet_hha_attention(num_classes, dropout=0.3, pretrained=True, encoder="resnet50"):
    print(f"Using Unet with {encoder}. Using dual encoders for RGB and HHA inputs.")
    return UnetAttentionHHA(num_classes=num_classes, dropout=dropout, pretrained=pretrained, encoder=encoder)
