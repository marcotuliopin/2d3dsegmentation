import torch
import torch.nn as nn
import torch.nn.functional as F


class SAM(nn.Module):
    """Spatial Attention Module (SAM) for enhancing feature maps with spatial attention.
    """
    def __init__(self, bias=False):
        super(SAM, self).__init__()
        self.bias = bias
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3, dilation=1, bias=self.bias)
        self.dropout = nn.Dropout2d(p=0.2)

    def forward(self, x):
        max = torch.max(x,1)[0].unsqueeze(1)
        avg = torch.mean(x,1).unsqueeze(1)
        concat = torch.cat((max,avg), dim=1)
        output = self.conv(concat)
        output = self.dropout(output)
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
            nn.Dropout(0.2),
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