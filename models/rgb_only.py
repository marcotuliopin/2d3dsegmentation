import torch
import torch.nn as nn

from models.resnet50 import ResNet50Decoder, ResNet50Encoder


class RBGOnly(nn.Module):
    def __init__(
        self,
        num_classes,
        dropout=0.3,
    ):
        super().__init__()

        self.encoder = ResNet50Encoder()
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

if __name__ == "__main__":
    # Example usage
    model = RBGOnly(num_classes=13)
    print(model)
    # Example input tensor with batch size 1, 3 channels (RGB), and 224x224 spatial dimensions
    example_input = torch.randn(1, 3, 224, 224)
    output = model(example_input)
    print(output.shape)  # Should match the expected output shape