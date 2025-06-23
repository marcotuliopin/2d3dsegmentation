import torch
import torch.nn as nn
from torchvision import models

class ResNet50Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        # We are using the ResNet50 model from torchvision to be able to easily load the pretrained weights
        self.encoder = models.resnet50(weights=models.resnet.ResNet50_Weights.DEFAULT)
        self.out_channels = (64, 256, 512, 1024, 2048)

    def forward(self, x):
        features = []
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        features.append(x)

        x = self.encoder.maxpool(x)

        x = self.encoder.layer1(x)
        features.append(x)

        x = self.encoder.layer2(x)
        features.append(x)

        x = self.encoder.layer3(x)
        features.append(x)

        x = self.encoder.layer4(x)
        features.append(x)

        return features


class ResNet50Decoder(nn.Module):
    def __init__(self, num_channels=14):
        super().__init__()

        self.in_channels = 2048

        ResBlock = BottleneckDecoder
        layer_list = [3, 4, 6, 3]

        self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=1024, stride=2)
        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=512, stride=2, skip_ch=1024)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=256, stride=2, skip_ch=512)
        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=128, stride=2, skip_ch=256)
        
        self.upsample = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.batch_norm_up = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        
        self.final_conv = nn.Conv2d(64, num_channels, kernel_size=1)
        
    def forward(self, x, skip_connections=None):
        x = self.layer4(x)
        
        if skip_connections is not None and len(skip_connections) >= 3:
            x = torch.cat([x, skip_connections[2]], dim=1)
            
        x = self.layer3(x)
        
        if skip_connections is not None and len(skip_connections) >= 2:
            x = torch.cat([x, skip_connections[1]], dim=1)
            
        x = self.layer2(x)

        if skip_connections is not None and len(skip_connections) >= 1:
            x = torch.cat([x, skip_connections[0]], dim=1)
            
        x = self.layer1(x)

        x = self.relu(self.batch_norm_up(self.upsample(x)))
        x = self.final_conv(x)
        
        return x
        
    def _make_layer(self, ResBlock, blocks, planes, stride=1, skip_ch=0):
        actual_in_channels = self.in_channels + skip_ch
        
        ii_upsample = None
        layers = []
        
        if stride != 1 or actual_in_channels != planes:
            ii_upsample = nn.Sequential(
                nn.ConvTranspose2d(actual_in_channels, planes, kernel_size=1, 
                                 stride=stride, output_padding=stride-1 if stride > 1 else 0),
                nn.BatchNorm2d(planes)
            )
            
        # First block handles the channel transition and potential skip connection
        layers.append(ResBlock(actual_in_channels, planes, i_upsample=ii_upsample, stride=stride))
        
        for _ in range(blocks-1):
            layers.append(ResBlock(planes, planes))
            
        self.in_channels = planes
        return nn.Sequential(*layers)


class BottleneckDecoder(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, i_upsample=None, stride=1, dropout=0.0):
        super(BottleneckDecoder, self).__init__()
        
        # Reverse the bottleneck: start with expanded channels, reduce to out_channels
        self.conv1 = nn.ConvTranspose2d(in_channels, in_channels//self.expansion, kernel_size=1, stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(in_channels//self.expansion)
        
        self.conv2 = nn.ConvTranspose2d(in_channels//self.expansion, in_channels//self.expansion, 
                                       kernel_size=3, stride=stride, padding=1, output_padding=stride-1 if stride > 1 else 0)
        self.batch_norm2 = nn.BatchNorm2d(in_channels//self.expansion)
        
        self.conv3 = nn.ConvTranspose2d(in_channels//self.expansion, out_channels, kernel_size=1, stride=1, padding=0)
        self.batch_norm3 = nn.BatchNorm2d(out_channels)

        self.dropout = nn.Dropout2d(p=dropout)
        
        self.i_upsample = i_upsample
        self.stride = stride
        self.relu = nn.ReLU()
        
    def forward(self, x):
        identity = x.clone()
        
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.dropout(x)

        x = self.relu(self.batch_norm2(self.conv2(x)))
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.batch_norm3(x)
        
        # Upsample identity if needed
        if self.i_upsample is not None:
            identity = self.i_upsample(identity)
            
        x += identity
        x = self.relu(x)
        
        return x


if __name__ == "__main__":
    model = ResNet50Decoder(num_channels=14)
    x = torch.randn(1, 2048, 7, 7)  # Simulate output from ResNet50
    skip_connections = [
        torch.randn(1, 256, 128, 128),  # Layer 1 skip connection
        torch.randn(1, 512, 64, 64),   # Layer 2 skip connection
        torch.randn(1, 1024, 32, 32)  # Layer 3 skip connection
    ]
    output = model(x, skip_connections=skip_connections)
    print("Decoder output shape:", output.shape)  # Should be [1, 3, 224, 224] if upsampled correctly