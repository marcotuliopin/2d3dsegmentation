import torch.nn as nn
import torchvision


def get_fcn_resnet50(num_classes=14, pretrained=False, dropout=0.5):
    if pretrained:
        from torchvision.models.segmentation import FCN_ResNet50_Weights
        model = torchvision.models.segmentation.fcn_resnet50(weights=FCN_ResNet50_Weights.DEFAULT)
    else:
        model = torchvision.models.segmentation.fcn_resnet50()
    
    model.classifier[4] = nn.Sequential(
        nn.Dropout(dropout),
        nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
    )
    return model