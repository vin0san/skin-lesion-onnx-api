import torch.nn as nn
import timm


class SkinLesionModel(nn.Module):
    
    def __init__(self, num_classes=9):
        super().__init__()
        self.model = timm.create_model('efficientnet_b3', pretrained=True, num_classes=num_classes)
    
    def forward(self, x):
        return self.model(x)