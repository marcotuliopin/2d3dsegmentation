import torch
import torch.nn as nn
import torch.nn.functional as F

from models.resnet50 import ResNet50Decoder, ResNet50Encoder


class AttentionLateFusion(nn.Module):
    def __init__(self, num_classes, dropout=0.3, d_channels=1):
        super().__init__()

        self.rgb_encoder = ResNet50Encoder()
        self.d_encoder = ResNet50Encoder(dropout=dropout)
        self._adapt_input_channels(d_channels)
        
        self.rgb_norms = nn.ModuleList([nn.BatchNorm2d(ch) for ch in self.rgb_encoder.out_channels])
        self.d_norms = nn.ModuleList([nn.BatchNorm2d(ch) for ch in self.d_encoder.out_channels])

        self.attention = nn.ModuleList([FRM(in_channels=ch, reduction_ratio=8) for ch in self.rgb_encoder.out_channels])

        self.decoder = ResNet50Decoder(num_channels=num_classes, dropout=dropout)

    def forward(self, x):
        rgb = x[:, :3, :, :]
        d = x[:, 3:, :, :]

        rgb_feats = self.rgb_encoder(rgb)
        d_feats = self.d_encoder(d)

        # Fusion of features using concatenation
        rgb_feats_norm = [norm(feat) for feat, norm in zip(rgb_feats, self.rgb_norms)]
        d_feats_norm = [norm(feat) for feat, norm in zip(d_feats, self.d_norms)]

        fused_feats = [self.attention[i](rgb_feats_norm[i], d_feats_norm[i]) for i in range(len(rgb_feats_norm))]
        x = self.decoder(fused_feats[-1], fused_feats[:-1])

        return x

    def get_optimizer_groups(self):
        rgb_encoder = [p for name, p in self.rgb_encoder.named_parameters() if "conv1" not in name]
        d_encoder = [p for name, p in self.d_encoder.named_parameters() if "conv1" not in name]

        decoder = list(self.decoder.parameters())
        attention = list(self.attention.parameters())

        return [
            {"params": self.rgb_encoder.conv1.parameters(), "lr": 5e-4},
            {"params": self.d_encoder.conv1.parameters(), "lr": 5e-3},
            {"params": rgb_encoder, "lr": 1e-4},
            {"params": d_encoder, "lr": 1e-3},
            {"params": self.rgb_norms.parameters(), "lr": 1e-3},
            {"params": self.d_norms.parameters(), "lr": 1e-3},
            {"params": attention, "lr": 1e-3},
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
                new_conv_d.weight.data = first_conv_rgb.weight.data.mean(dim=1, keepdim=True)
            else:
                new_conv_d.weight.data = first_conv_rgb.weight.data.clone()
            if first_conv_d.bias is not None:
                new_conv_d.bias.data = first_conv_d.bias.data.clone()

        self.d_encoder.encoder.conv1 = new_conv_d


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
    Channel Attention module (CAM) for cross-attention between RGB and X features.
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
    Convolutional Block Attention Module (CBAM) for cross-attention between RGB and X features.
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
    Fusion Refinement Module (FRM) for cross-attention between RGB and X features.
    """
    def __init__(self, in_channels, reduction_ratio=8):
        super().__init__()
        self.rgb_att = CBAM(in_channels, reduction_ratio)
        self.d_att = CBAM(in_channels, reduction_ratio)

    def forward(self, rgb_feat, d_feat):
        rgb_ref = self.rgb_att(rgb_feat)
        d_ref = self.d_att(d_feat)
        output = rgb_ref + d_ref
        return output


if __name__ == "__main__":
    # Example usage
    model = AttentionLateFusion(num_classes=13, d_channels=1)
    print(model)
    
    # Test the forward pass with a dummy input
    dummy_input = torch.randn(1, 4, 224, 224)  # Batch size of 1, 4 channels (RGB + Depth), 224x224 image
    output = model(dummy_input)
    print(output.shape)  # Should be [1, num_classes, H, W]