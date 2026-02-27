import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights

from config import DEVICE


def build_model(num_classes):
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(DEVICE)


def unfreeze_last_blocks(model):
    for name, param in model.named_parameters():
        if any(block in name for block in ("layer3", "layer4", "fc")):
            param.requires_grad = True
