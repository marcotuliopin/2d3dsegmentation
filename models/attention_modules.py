import torch
import torch.nn as nn
import torch.nn.functional as F


class CMA(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8):
        super().__init__()
        self.reduced_dim = in_channels // reduction_ratio
        if self.reduced_dim < 1:
            self.reduced_dim = in_channels
        
        self.cross_conv = nn.Conv2d(in_channels * 2, self.reduced_dim, 1, bias=False)
        self.bn = nn.BatchNorm2d(self.reduced_dim)
        self.relu = nn.ReLU(inplace=True)
        
        self.fc = nn.Conv2d(self.reduced_dim, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, y):
        x_pool = F.adaptive_avg_pool2d(x, 1)
        y_pool = F.adaptive_avg_pool2d(y, 1)
        
        cross_feat = torch.cat([x_pool, y_pool], dim=1)
        cross_feat = self.relu(self.bn(self.cross_conv(cross_feat)))
        
        att_weights = self.sigmoid(self.fc(cross_feat))
        
        return x * att_weights


class DoubleCMA(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8):
        super().__init__()
        
        self.cma_xy = CMA(in_channels, reduction_ratio)
        self.cma_yx = CMA(in_channels, reduction_ratio)
        
    def forward(self, x, y):
        return self.cma_xy(x, y), self.cma_yx(y, x)


class CMSA(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8):
        super().__init__()
        self.in_channels = in_channels
        self.reduced_dim = in_channels // reduction_ratio
        if self.reduced_dim < 1:
            self.reduced_dim = in_channels
        
        self.query_x = nn.Conv2d(in_channels, self.reduced_dim, 1)
        self.key_x = nn.Conv2d(in_channels, self.reduced_dim, 1)
        self.value_x = nn.Conv2d(in_channels, in_channels, 1)
        
        self.query_y = nn.Conv2d(in_channels, self.reduced_dim, 1)
        self.key_y = nn.Conv2d(in_channels, self.reduced_dim, 1)
        self.value_y = nn.Conv2d(in_channels, in_channels, 1)
        
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x_features, y_features):
        B, C, H, W = x_features.size()
        
        q_x = self.query_x(x_features).view(B, self.reduced_dim, H * W)     # (B, C', HW)
        k_x = self.key_x(x_features).view(B, self.reduced_dim, H * W)       # (B, C', HW)
        v_x = self.value_x(x_features).view(B, C, H * W)                    # (B, C, HW)
        
        q_y = self.query_y(y_features).view(B, self.reduced_dim, H * W) # (B, C', HW)
        k_y = self.key_y(y_features).view(B, self.reduced_dim, H * W)   # (B, C', HW)
        v_y = self.value_y(y_features).view(B, C, H * W)                # (B, C, HW)

        attention_y_to_x = torch.bmm(q_y.permute(0, 2, 1), k_y)  # (B, HW, HW)
        attention_y_to_x = self.softmax(attention_y_to_x)

        enhanced_x = torch.bmm(v_x, attention_y_to_x.permute(0, 2, 1))  # (B, C, HW)
        enhanced_x = enhanced_x.view(B, C, H, W) + x_features

        attention_x_to_y = torch.bmm(q_x.permute(0, 2, 1), k_x)  # (B, HW, HW)
        attention_x_to_y = self.softmax(attention_x_to_y)

        enhanced_y = torch.bmm(v_y, attention_x_to_y.permute(0, 2, 1))  # (B, C, HW)
        enhanced_y = enhanced_y.view(B, C, H, W) + y_features
        
        return enhanced_x, enhanced_y


class GC_FFM(nn.Module):
    def __init__(self, in_channels, spatial_threshold=128*128):
        super().__init__()
        
        self.spatial_threshold = spatial_threshold

        self.cross_attention = CMSA(in_channels)
        
        self.refine_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 1),
        )

        self.residual_conv = nn.Conv2d(in_channels * 2, in_channels, 1)
        self.normalize = nn.BatchNorm2d(in_channels)

    def forward(self, x, y):
        _, _, H, W = x.size()
        spatial_size = H * W

        if spatial_size > self.spatial_threshold:
            downsample_factor = int((spatial_size / self.spatial_threshold) ** 0.5) + 1
            x_downsampled = F.avg_pool2d(x, downsample_factor, downsample_factor)
            y_downsampled = F.avg_pool2d(y, downsample_factor, downsample_factor)
            x_attended, y_attended = self.cross_attention(x_downsampled, y_downsampled)

            x_attended = F.interpolate(x_attended, size=(H, W), mode='bilinear', align_corners=False)
            y_attended = F.interpolate(y_attended, size=(H, W), mode='bilinear', align_corners=False)
        else:
            x_attended, y_attended = self.cross_attention(x, y)
        
        fused_feat = torch.cat([x_attended, y_attended], dim=1)

        refined_feat = self.refine_conv(fused_feat)
        residual = self.residual_conv(fused_feat)

        refined_feat = refined_feat + residual
        refined_feat = self.normalize(refined_feat)
        
        return refined_feat


class SA_FM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8):
        super().__init__()
        reduced_dim = in_channels // reduction_ratio
        if reduced_dim < 1:
            reduced_dim = in_channels

        self.x_to_y_attention = nn.Sequential(
            nn.Conv2d(in_channels * 2, reduced_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_dim, 1, 1),
            nn.Sigmoid()
        )
        
        self.y_to_x_attention = nn.Sequential(
            nn.Conv2d(in_channels * 2, reduced_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_dim, 1, 1),
            nn.Sigmoid()
        )
        
        self.refine_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 1),
        )
        
        self.residual_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1),
        )
        self.normalize = nn.BatchNorm2d(in_channels)
        
    def forward(self, x, y):
        joint_features = torch.cat([x, y], dim=1)

        att_x_to_y = self.x_to_y_attention(joint_features)
        att_y_to_x = self.y_to_x_attention(joint_features)
        
        x_enhanced = x * att_y_to_x
        y_enhanced = y * att_x_to_y
        
        fused = torch.cat([x_enhanced, y_enhanced], dim=1)
        output = self.refine_conv(fused)
        residual = self.residual_conv(fused)
        output = output + residual
        output = self.normalize(output)

        return output