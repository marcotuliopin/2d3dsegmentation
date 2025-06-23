import torch
import torch.nn as nn

class UNetDecoder(nn.Module):
    def __init__(self, encoder_channels=[64, 256, 512, 1024, 2048], num_classes=40):
        super(UNetDecoder, self).__init__()
        
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(encoder_channels[4], 512, 2, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(512 + encoder_channels[3], 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(256 + encoder_channels[2], 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128 + encoder_channels[1], 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64 + encoder_channels[0], 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.up5 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.pred1 = nn.Conv2d(512, num_classes, 1)
        self.pred2 = nn.Conv2d(256, num_classes, 1)
        self.pred3 = nn.Conv2d(128, num_classes, 1)
        self.pred4 = nn.Conv2d(64, num_classes, 1)
        
        self.final_conv = nn.Conv2d(32, num_classes, 1)
    
    def forward(self, encoder_features):
        outputs = []
        
        # Start from bottleneck (feat5)
        x = encoder_features[4]  # (N, 2048, H/32, W/32)
        
        # Decoder stage 1: bottleneck -> feat4 level
        x = self.up1(x)  # (N, 512, H/16, W/16) - upsample bottleneck
        x = torch.cat([x, encoder_features[3]], dim=1)  # Skip connection with feat4
        x = self.conv1(x)  # Process combined features
        if self.training:
            outputs.append(self.pred1(x))  # Multi-scale output 1
        
        # Decoder stage 2: -> feat3 level  
        x = self.up2(x)  # (N, 256, H/8, W/8)
        x = torch.cat([x, encoder_features[2]], dim=1)  # Skip connection with feat3
        x = self.conv2(x)
        if self.training:
            outputs.append(self.pred2(x))  # Multi-scale output 2
        
        # Decoder stage 3: -> feat2 level
        x = self.up3(x)  # (N, 128, H/4, W/4)
        x = torch.cat([x, encoder_features[1]], dim=1)  # Skip connection with feat2
        x = self.conv3(x)
        if self.training:
            outputs.append(self.pred3(x))  # Multi-scale output 3
        
        # Decoder stage 4: -> feat1 level
        x = self.up4(x)  # (N, 64, H/2, W/2)
        x = torch.cat([x, encoder_features[0]], dim=1)  # Skip connection with feat1
        x = self.conv4(x)
        if self.training:
            outputs.append(self.pred4(x))  # Multi-scale output 4
        
        # Final prediction at feat1 resolution
        x = self.up5(x)
        final_pred = self.final_conv(x)  # (N, num_classes, H/2, W/2)
        outputs.append(final_pred)
        
        return outputs
