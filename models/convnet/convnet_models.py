import torch.nn as nn
import torchvision


def vgg16(num_classes):
    model = torchvision.models.vgg16(pretrained=False, progress=True)
    model.classifier[6] = nn.Linear(in_features=4096, out_features=num_classes, bias=True)
    return model
