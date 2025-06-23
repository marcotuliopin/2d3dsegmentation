import torch
import torch.nn as nn

from models.resnet50 import ResNet50Encoder
from models.unet import UNetDecoder


class LateFusion(nn.Module):
    def __init__(self, num_classes, dropout=0.3, d_channels=1):
        super().__init__()

        self.rgb_encoder = ResNet50Encoder()
        self.d_encoder = ResNet50Encoder()
        self._adapt_input_channels(d_channels)
        
        self.rgb_norms = nn.ModuleList([nn.BatchNorm2d(ch) for ch in self.rgb_encoder.out_channels])
        self.d_norms = nn.ModuleList([nn.BatchNorm2d(ch) for ch in self.d_encoder.out_channels])

        self.decoder = UNetDecoder(encoder_channels=self.d_encoder.out_channels, num_classes=num_classes)

    def forward(self, x):
        rgb = x[:, :3, :, :]
        d = x[:, 3:, :, :]

        rgb_feats = self.rgb_encoder(rgb)
        d_feats = self.d_encoder(d)

        rgb_feats_norm = [norm(feat) for feat, norm in zip(rgb_feats, self.rgb_norms)]
        d_feats_norm = [norm(feat) for feat, norm in zip(d_feats, self.d_norms)]
        x = [r + h for r, h in zip(rgb_feats_norm, d_feats_norm)]

        x = self.decoder(x)

        return x
    
    def get_optimizer_groups(self):
        attention_parameters = list(self.rgb_norms.parameters()) + list(self.d_norms.parameters())

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
    model = LateFusion(num_classes=13, d_channels=1)
    print(model)
    
    # Example input tensor with batch size 1, 4 channels (RGB + Depth), and 224x224 spatial dimensions
    example_input = torch.randn(1, 4, 224, 224)
    output = model(example_input)
    print([out.shape for out in output])  # Should match the expected output shape