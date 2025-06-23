import torch
import torch.nn as nn

from models.attention_modules import DoubleCMA
from models.resnet50 import ResNet50Encoder
from models.unet import UNetDecoder


class AttentionMidFusion(nn.Module):
    def __init__(self, num_classes, dropout=0.3, d_channels=3):
        super().__init__()

        self.rgb_encoder = ResNet50Encoder()
        self.d_encoder = ResNet50Encoder()
        self._adapt_input_channels(d_channels)

        self.modulator = nn.ModuleList([
            DoubleCMA(in_channels=64, reduction_ratio=8),
            DoubleCMA(in_channels=256, reduction_ratio=8),
            DoubleCMA(in_channels=512, reduction_ratio=8),
            DoubleCMA(in_channels=1024, reduction_ratio=8),
            DoubleCMA(in_channels=2048, reduction_ratio=8)
        ])

        self.decoder = UNetDecoder(encoder_channels=self.d_encoder.out_channels, num_classes=num_classes)

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

        rgb, d = self.modulator[0](rgb, d)

        rgb_feats.append(rgb)
        d_feats.append(d)

        rgb = self.rgb_encoder.encoder.maxpool(rgb)
        d = self.d_encoder.encoder.maxpool(d)

        rgb = self.rgb_encoder.encoder.layer1(rgb)
        d = self.d_encoder.encoder.layer1(d)

        rgb, d = self.modulator[1](rgb, d)

        rgb_feats.append(rgb)
        d_feats.append(d)

        rgb = self.rgb_encoder.encoder.layer2(rgb)
        d = self.d_encoder.encoder.layer2(d)

        rgb, d = self.modulator[2](rgb, d)

        rgb_feats.append(rgb)
        d_feats.append(d)

        rgb = self.rgb_encoder.encoder.layer3(rgb)
        d = self.d_encoder.encoder.layer3(d)

        rgb, d = self.modulator[3](rgb, d)

        rgb_feats.append(rgb)
        d_feats.append(d)

        rgb = self.rgb_encoder.encoder.layer4(rgb)
        d = self.d_encoder.encoder.layer4(d)

        rgb, d = self.modulator[4](rgb, d)

        rgb_feats.append(rgb)
        d_feats.append(d)
        
        x = [r + h for r, h in zip(rgb_feats, d_feats)]
        x = self.decoder(x)

        return x
    
    def get_optimizer_groups(self):
        attention_parameters = list(self.modulator.parameters())

        return [
            {"params": self.rgb_encoder.encoder.parameters(), "lr": 1e-4},
            {"params": self.d_encoder.encoder.parameters(), "lr": 3e-4},
            {"params": attention_parameters, "lr": 3e-4},
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
                # Average the RGB channels to initialize the depth channel
                new_conv_d.weight.data = first_conv_rgb.weight.data.mean(dim=1, keepdim=True)
            else:
                new_conv_d.weight.data = first_conv_rgb.weight.data.clone()
            if first_conv_d.bias is not None:
                new_conv_d.bias.data = first_conv_d.bias.data.clone()

        self.d_encoder.encoder.conv1 = new_conv_d


if __name__ == "__main__":
    # Example usage
    model = AttentionMidFusion(num_classes=13, d_channels=3)
    print(model)
    
    # Test the model with dummy input
    dummy_input = torch.randn(1, 6, 224, 224)  # Batch size of 1, 6 channels (3 RGB + 3 Depth)
    output = model(dummy_input)
    print([out.shape for out in output])  # Should match the expected output shape
