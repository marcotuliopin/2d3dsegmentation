import torch.nn as nn
import torchvision


def get_deeplabv3_resnet50(num_classes, pretrained=False, dropout=0.5):
    if pretrained:
        from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights
        model = torchvision.models.segmentation.deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
    else:
        model = torchvision.models.segmentation.deeplabv3_resnet50()
    
    model.classifier[4] = nn.Sequential(
        nn.Dropout(dropout),
        nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
    )
    return model