import torch
import torch.nn as nn
import torch.nn.functional as F

import segmentation_models_pytorch as smp


class UNetAttEarlyFusionHHA(nn.Module):
    def __init__(
        self,
        num_classes,
        encoder="resnet50",
        dropout=0.3,
        pretrained=True,
    ):
        super().__init__()

        weights = "imagenet" if pretrained else None

        self.unet = smp.Unet(
            encoder_name=encoder,
            encoder_weights=weights,
            in_channels=3,
            classes=num_classes,
            activation=None,
        )

        # Modify to accept 6 input channels (3 RGB + 3 HHA)
        if pretrained:
            self._adapt_input_channels()
        else:
            self.unet = smp.Unet(
                encoder_name=encoder,
                encoder_weights=None,
                in_channels=6,
                classes=num_classes,
                activation=None,
            )

        self.fuse1 = CBAM(in_channels=self.unet.encoder.out_channels[1], reduction_ratio=8)
        self.fuse2 = CBAM(in_channels=self.unet.encoder.out_channels[2], reduction_ratio=8)
        self.fuse3 = CBAM(in_channels=self.unet.encoder.out_channels[3], reduction_ratio=8)
        self.fuse4 = CBAM(in_channels=self.unet.encoder.out_channels[4], reduction_ratio=8)
        self.fuse5 = CBAM(in_channels=self.unet.encoder.out_channels[5], reduction_ratio=8)

        # Add dropout in all decoder blocks
        for i in range(len(self.unet.decoder.blocks)):
            self.unet.decoder.blocks[i].dropout = nn.Dropout2d(p=dropout)

    def forward(self, x):
        x1 = self.unet.encoder.conv1(x)
        x1 = self.unet.encoder.maxpool(x1)
        x1 = self.fuse1(x1)
        x2 = self.unet.encoder.layer1(x1)
        x2 = self.fuse2(x2)
        x3 = self.unet.encoder.layer2(x2)
        x3 = self.fuse3(x3)
        x4 = self.unet.encoder.layer3(x3)
        x4 = self.fuse4(x4)
        x5 = self.unet.encoder.layer4(x4)
        x5 = self.fuse5(x5)
        x = self.unet.decoder([x, x1, x2, x3, x4, x5])
        x = self.unet.segmentation_head(x)
        return x

    def set_trainable(self, trainable=True):
        for param in self.unet.encoder.parameters():
            param.requires_grad = trainable

    def get_optimizer_groups(self):
        first_layer = list(self.unet.encoder.conv1.parameters())
        encoder = [p for name, p in self.unet.encoder.named_parameters() if "conv1" not in name]
        decoder = list(self.unet.decoder.parameters()) + list(self.unet.segmentation_head.parameters())

        return [
            {"params": first_layer, "lr": 5e-3},        
            {"params": encoder, "lr": 1e-3},
            {"params": decoder, "lr": 1e-2},
        ]

    def _adapt_input_channels(self):
        first_conv = self.unet.encoder.conv1  # For Resnet

        new_conv = nn.Conv2d(
            6, 
            first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=first_conv.bias is not None
        )

        with torch.no_grad():
            # Copy original RGB weights
            new_conv.weight.data[:, :3, :, :] = first_conv.weight.data
            # Duplicate the RGB weights for HHA channels
            new_conv.weight.data[:, 3:, :, :] = first_conv.weight.data

            if first_conv.bias is not None:
                new_conv.bias.data = first_conv.bias.data.clone()

        self.unet.encoder.conv1 = new_conv


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


def get_unet_early_fusion_hha_att(num_classes, dropout=0.3, pretrained=True, encoder="resnet50"):
    print(f"Using model: UNet with {encoder} encoder. HHA concatenation enabled.")
    return UNetAttEarlyFusionHHA(
        num_classes=num_classes,
        dropout=dropout,
        pretrained=pretrained,
        encoder=encoder,
    )
