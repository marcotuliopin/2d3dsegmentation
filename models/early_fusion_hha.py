import torch
import torch.nn as nn

from models.resnet50 import ResNet50Decoder, ResNet50Encoder


class EarlyFusionHHA(nn.Module):
    def __init__(
        self,
        num_classes,
        dropout=0.3,
    ):
        super().__init__()

        self.encoder = ResNet50Encoder()
        self._adapt_input_channels()

        self.decoder = ResNet50Decoder(num_channels=num_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x[-1], x[:-1])
        return x

    def get_optimizer_groups(self):
        encoder = list(self.encoder.encoder.parameters())
        decoder = list(self.decoder.parameters())

        return [
            {"params": encoder, "lr": 1e-4},
            {"params": decoder, "lr": 5e-4},
        ]

    def _adapt_input_channels(self):
        first_conv = self.encoder.encoder.conv1  # For Resnet

        new_conv = nn.Conv2d(
            6, 
            first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=first_conv.bias is not None
        )

        with torch.no_grad():
            new_conv.weight.data[:, :3, :, :] = first_conv.weight.data
            new_conv.weight.data[:, 3:, :, :] = first_conv.weight.data
            if first_conv.bias is not None:
                new_conv.bias.data = first_conv.bias.data.clone()

        self.encoder.encoder.conv1 = new_conv


if __name__ == "__main__":
    # Example usage
    model = EarlyFusionHHA(num_classes=13)
    print(model)
    
    # Create a dummy input tensor with 6 channels (3 RGB + 3 HHA)
    dummy_input = torch.randn(1, 6, 224, 224)
    
    # Forward pass
    output = model(dummy_input)
    print(output.shape)  # Should be [1, num_classes, H, W] where H and W depend on the input size
