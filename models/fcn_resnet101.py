import torch.nn as nn
import torchvision


def get_fcn_resnet101(num_classes, pretrained=False, dropout=0.5):
    print("Using model: FCN ResNet101")
    if pretrained:
        from torchvision.models.segmentation import FCN_ResNet101_Weights
        model = torchvision.models.segmentation.fcn_resnet101(weights=FCN_ResNet101_Weights.DEFAULT)
    else:
        model = torchvision.models.segmentation.fcn_resnet101()
    
    model.classifier[4] = nn.Sequential(
        nn.Dropout(dropout),
        nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
    )
    return model