import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder


class UnetAttentionHHA(nn.Module):
    def __init__(self, num_classes, encoder="resnet50", dropout=0.3, pretrained=True, attention_reduction_ratio=8):
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
        
        # One attention module for every layer of the encoder
        self.fusion = nn.ModuleList([
            CrossAttentionFusion(in_channels=ch, reduction_ratio=attention_reduction_ratio) 
            for ch in self.rgb_encoder.out_channels
        ]) 

        self.encoder_channels = self.rgb_encoder.out_channels

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

        rgb_feats_norm = [norm(feat) for feat, norm in zip(rgb_feats, self.rgb_norms)]
        hha_feats_norm = [norm(feat) for feat, norm in zip(hha_feats, self.hha_norms)]

        # Fusion of features using concatenation
        fused_feats = []
        for rgb_feat, hha_feat, fusion in zip(rgb_feats_norm, hha_feats_norm, self.fusion):
            fused = fusion(rgb_feat, hha_feat)
            fused_feats.append(fused)
        
        decoder_output = self.decoder(fused_feats)
        dropout_output = self.dropout(decoder_output)
        return self.segmentation_head(dropout_output)

    
    def get_optimizer_groups(self):
        first_layer_rgb = list(self.rgb_encoder.conv1.parameters())
        first_layer_hha = list(self.hha_encoder.conv1.parameters())

        rgb_encoder = list(p for name, p in self.rgb_encoder.named_parameters() if "conv1" not in name)
        hha_encoder = list(p for name, p in self.hha_encoder.named_parameters() if "conv1" not in name)

        attention_params = list(self.fusion.parameters())

        decoder = list(self.decoder.parameters()) + list(self.segmentation_head.parameters())

        return [
            {"params": first_layer_rgb, "lr": 5e-4},
            {"params": first_layer_hha, "lr": 5e-3},
            {"params": rgb_encoder, "lr": 1e-4},
            {"params": hha_encoder, "lr": 1e-3},
            {"params": attention_params, "lr": 1e-3},
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


class CrossAttentionFusion(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8):
        super().__init__()
        self.reduced_dim = max(in_channels // reduction_ratio, 16)

        # RGB attends to HHA
        self.rgb_to_q = nn.Conv2d(in_channels, self.reduced_dim, 1)
        self.hha_to_k = nn.Conv2d(in_channels, self.reduced_dim, 1)
        self.hha_to_v = nn.Conv2d(in_channels, self.reduced_dim, 1)

        # Reverse attention (HHA → RGB)
        self.hha_to_q = nn.Conv2d(in_channels, self.reduced_dim, 1)
        self.rgb_to_k = nn.Conv2d(in_channels, self.reduced_dim, 1)
        self.rgb_to_v = nn.Conv2d(in_channels, self.reduced_dim, 1)

        # We are currently using concatenation as the fusion method
        # TODO: Explore more fusion methods like sum or gated
        self.output_proj = nn.Conv2d(self.reduced_dim * 2, in_channels, 1)
        self.norm = nn.BatchNorm2d(in_channels)

        # Add a residual connection to help with gradient flow
        self.residual_proj = nn.Conv2d(in_channels, in_channels, 1)
        self.attention_weight = nn.Parameter(torch.tensor(0.1))

    def cross_attention(self, q, k, v, pool_hw=32):
        B, C, H, W = q.shape

        # Reduce spatial size to fixed value to avoid huge attention maps
        q = F.adaptive_avg_pool2d(q, (pool_hw, pool_hw))
        k = F.adaptive_avg_pool2d(k, (pool_hw, pool_hw))
        v = F.adaptive_avg_pool2d(v, (pool_hw, pool_hw))

        q = q.view(B, C, -1).transpose(1, 2)  # (B, HW, C)
        k = k.view(B, C, -1)                  # (B, C, HW)
        v = v.view(B, C, -1).transpose(1, 2)  # (B, HW, C)

        attn_scores = torch.bmm(q, k) / math.sqrt(C)
        attn = F.softmax(attn_scores, dim=-1)
        out = torch.bmm(attn, v).transpose(1, 2).view(B, C, pool_hw, pool_hw)

        # Upsample back to original resolution
        return F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)

    def forward(self, rgb_feat, hha_feat):
        # RGB attends to HHA
        q1 = self.rgb_to_q(rgb_feat)
        k1 = self.hha_to_k(hha_feat)
        v1 = self.hha_to_v(hha_feat)
        rgb_att = self.cross_attention(q1, k1, v1)

        # HHA attends to RGB
        q2 = self.hha_to_q(hha_feat)
        k2 = self.rgb_to_k(rgb_feat)
        v2 = self.rgb_to_v(rgb_feat)
        hha_att = self.cross_attention(q2, k2, v2)

        fused = torch.cat([rgb_att, hha_att], dim=1)
        attention_output = self.output_proj(fused)

        # Without residual connection, gradients may not propagate well
        # across the network, especially in deeper layers.
        # Adding residual connection to help with gradient flow.
        residual = self.residual_proj(rgb_feat) # Using only RGB feature for residual to avoid issues with modal fusion
        output = self.attention_weight * attention_output + residual

        return self.norm(output)


def get_unet_hha_attention(
    num_classes,
    dropout=0.3,
    pretrained=True,
    encoder="resnet50",
    attention_reduction_ratio=8,
):
    print(f"Using Unet with {encoder}. Using cross-atention to fuse RGB and HHA.")
    return UnetAttentionHHA(
        num_classes=num_classes,
        dropout=dropout,
        pretrained=pretrained,
        encoder=encoder,
        attention_reduction_ratio=attention_reduction_ratio,
    )
