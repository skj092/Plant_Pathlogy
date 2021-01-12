import torchvision
import torch.nn as nn
import torch.nn.functional as F

class PlantModel(nn.Module):
    
    def __init__(self, num_classes):
        super().__init__()
        
        self.backbone = torchvision.models.resnet18(pretrained=True)
        
        in_features = self.backbone.fc.in_features
        
        self.logit = nn.ModuleList(
            [nn.Linear(in_features, c) for c in num_classes]
        )
        
    def forward(self, x):
        batch_size, C, H, W = x.shape
        
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        x = F.adaptive_avg_pool2d(x,1).reshape(batch_size,-1)
        x = F.dropout(x, 0.25, self.training)

        logit = [l(x) for l in self.logit]

        return logit