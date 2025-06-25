import torch
import torch.nn as nn

from models.attention_modules import SA_FM
from models.resnet50 import ResNet50Encoder
from models.unet import UNetDecoder


class AttentionLateFusion(nn.Module):
    def __init__(self, num_classes, dropout=0.3, d_channels=1):
        super().__init__()

        self.rgb_encoder = ResNet50Encoder()
        self.d_encoder = ResNet50Encoder()
        self._adapt_input_channels(d_channels)
        
        self.attention = nn.ModuleList([
            SA_FM(in_channels=64),
            SA_FM(in_channels=256),
            SA_FM(in_channels=512),
            SA_FM(in_channels=1024),
            SA_FM(in_channels=2048)
        ])

        self.decoder = UNetDecoder(encoder_channels=self.d_encoder.out_channels, num_classes=num_classes)

        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        rgb = x[:, :3, :, :]
        d = x[:, 3:, :, :]

        rgb_feats = []
        d_feats = []

        rgb = self.rgb_encoder.encoder.conv1(rgb)
        rgb = self.rgb_encoder.encoder.bn1(rgb)
        rgb = self.rgb_encoder.encoder.relu(rgb)

        d = self.d_encoder.encoder.conv1(d)
        d = self.d_encoder.encoder.bn1(d)
        d = self.d_encoder.encoder.relu(d)

        rgb_feats.append(rgb)
        d_feats.append(d)

        rgb = self.rgb_encoder.encoder.maxpool(rgb)
        d = self.d_encoder.encoder.maxpool(d)

        rgb = self.rgb_encoder.encoder.layer1(rgb)
        d = self.d_encoder.encoder.layer1(d)

        rgb_feats.append(rgb)
        d_feats.append(d)

        rgb = self.rgb_encoder.encoder.layer2(rgb)
        d = self.d_encoder.encoder.layer2(d)

        rgb_feats.append(rgb)
        d_feats.append(d)

        rgb = self.rgb_encoder.encoder.layer3(rgb)
        d = self.d_encoder.encoder.layer3(d)

        rgb_feats.append(rgb)
        d_feats.append(d)

        rgb = self.rgb_encoder.encoder.layer4(rgb)
        d = self.d_encoder.encoder.layer4(d)

        rgb_feats.append(rgb)
        d_feats.append(d)

        fused_feats = []
        for i in range(len(self.attention)):
            rgb_feat = rgb_feats[i]
            d_feat = d_feats[i]

            fused_feat = self.attention[i](rgb_feat, d_feat)
            fused_feats.append(fused_feat)

        x = self.decoder(fused_feats)

        return x

    def get_optimizer_groups(self):
        return [
            {"params": self.rgb_encoder.encoder.parameters(), "lr": 1e-4},
            {"params": self.d_encoder.encoder.parameters(), "lr": 3e-4},
            {"params": self.attention.parameters(), "lr": 3e-4},
            {"params": self.decoder.parameters(), "lr": 5e-4},
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


if __name__ == "__main__":
    # Example usage
    model = AttentionLateFusion(num_classes=13, d_channels=1)
    print(model)
    
    # Test the forward pass with a dummy input
    dummy_input = torch.randn(1, 4, 224, 224)  # Batch size of 1, 4 channels (RGB + Depth), 224x224 image
    output = model(dummy_input)
    print([out.shape for out in output])  # Should be [1, num_classes, H, W]