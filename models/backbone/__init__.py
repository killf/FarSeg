from .resnet import *

BACKBONES = {
    "ResNet18": resnet18,
    "ResNet34": resnet34,
    "ResNet50": resnet50,
    "ResNet101": resnet101,
    "ResNet152": resnet152,
    "ResNext50_32x4d": resnext50_32x4d,
    "ResNeXt101_32x8d": resnext101_32x8d,
    "WideResNet50_2": wide_resnet50_2,
    "WideResNet101_2": wide_resnet101_2
}
