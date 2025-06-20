import torch
import torch.nn as nn
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder

from models import resnet


class UnetMidFusionHHA(nn.Module):
    def __init__(self, num_classes, encoder="resnet50", dropout=0.3, pretrained=True):
        super().__init__()

        rgb_weights = "imagenet" if pretrained else None

        # Encoder RGB
        # Output channels are [3, 64, 256, 512, 1024, 2048] for resnet50
        self.rgb_encoder = get_encoder(
            name=encoder,
            in_channels=3,
            dropout=dropout,
            weights=rgb_weights
        )

        # Encoder HHA
        self.hha_encoder = get_encoder(
            name=encoder,
            in_channels=3,
            dropout=dropout,
            weights=None
        )

        if pretrained:
            self._adapt_input_channels()

        self.fusion_encoder = Encoder(dropout=dropout)
        
        self.rgb_norms = nn.ModuleList([nn.GroupNorm(1, ch) for ch in self.rgb_encoder.out_channels ])
        self.hha_norms = nn.ModuleList([nn.GroupNorm(1, ch) for ch in self.hha_encoder.out_channels ])

        self.decoder = UnetDecoder(
            encoder_channels=self.fusion_encoder.out_channels,
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

        rgb_feats_norm = [norm(feat) for feat, norm in zip(rgb_feats, self.rgb_norms)]
        hha_feats_norm = [norm(feat) for feat, norm in zip(hha_feats, self.hha_norms)]

        fused0 = rgb_feats_norm[1] + hha_feats_norm[1]

        fused1 = self.fusion_encoder.layer1(fused0)
        fused1 += rgb_feats_norm[2] + hha_feats_norm[2]

        fused2 = self.fusion_encoder.layer2(fused1)
        fused2 += rgb_feats_norm[3] + hha_feats_norm[3]

        fused3 = self.fusion_encoder.layer3(fused2)
        fused3 += rgb_feats_norm[4] + hha_feats_norm[4]

        fused4 = self.fusion_encoder.layer4(fused3)
        fused4 += rgb_feats_norm[5] + hha_feats_norm[5]


        output = self.decoder([rgb, fused0, fused1, fused2, fused3, fused4])
        output = self.dropout(output) 
        return self.segmentation_head(output)
    
    def get_optimizer_groups(self):
        rgb_encoder = [p for name, p in self.rgb_encoder.named_parameters() if "conv1" not in name]
        hha_encoder = [p for name, p in self.hha_encoder.named_parameters() if "conv1" not in name]
        fusion_encoder = list(self.fusion_encoder.parameters())

        decoder = list(self.decoder.parameters()) + list(self.segmentation_head.parameters())

        return [
            {"params": self.rgb_encoder.conv1.parameters(), "lr": 5e-4},
            {"params": self.hha_encoder.conv1.parameters(), "lr": 5e-3},
            {"params": rgb_encoder, "lr": 1e-4},
            {"params": hha_encoder, "lr": 1e-3},
            {"params": fusion_encoder, "lr": 1e-3},
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
    
    def _create_bottleneck(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dropout=0.0):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()
        )
    

class Encoder(nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        self.layer1 = resnet.make_layer(resnet.Bottleneck, 64, blocks=3, planes=64, stride=2)
        self.layer2 = resnet.make_layer(resnet.Bottleneck, 256, blocks=4, planes=128, stride=2)
        self.layer3 = resnet.make_layer(resnet.Bottleneck, 512, blocks=6, planes=256, stride=2)
        self.layer4 = resnet.make_layer(resnet.Bottleneck, 1024, blocks=3, planes=512, stride=2)
        self.dropout = nn.Dropout2d(p=dropout)

        self.out_channels = [3, 64, 256, 512, 1024]
    
    def forward(self, x):
        out = [x]
        x = self.layer1(x)
        out.append(x)
        x = self.layer2(x)
        out.append(x)
        x = self.layer3(x)
        out.append(x)
        x = self.layer4(x)
        out.append(x)
        x = self.layer5(x)
        x = self.dropout(x)
        out.append(x)
        return out


def get_unet_mid_fusion_hha(num_classes, dropout=0.3, pretrained=True, encoder="resnet50"):
    print(f"Using Unet with {encoder}. Using dual encoders for RGB and HHA inputs.")
    return UnetMidFusionHHA(num_classes=num_classes, dropout=dropout, pretrained=pretrained, encoder=encoder)


if __name__ == "__main__":
    enc = Encoder()
    x = torch.randn(1, 64, 128, 128)  # Batch size of 1, 6 channels (3 RGB + 3 HHA)
    out = enc(x)
    for i, o in enumerate(out):
        print(f"Output of layer {i}: {o.shape}")