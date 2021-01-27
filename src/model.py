import pretrainedmodels
import torch.nn as nn

def PlantModel(pretrained):
    if pretrained:
        model = pretrainedmodels.__dict__['resnet18'](
            pretrained='imagenet')
    else:
        model = pretrainedmodels.__dict__['resnet18'](
            pretrained=None)
    model.last_linear = nn.Sequential(
        nn.Linear(in_features=512, out_features=4),
        nn.Sigmoid())
    return model