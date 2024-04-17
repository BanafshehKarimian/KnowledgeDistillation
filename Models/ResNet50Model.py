from torchvision import models
import torch
import torch.optim as optim
import torch.nn as nn
from datasets import load_dataset
from PIL import Image 
from torchvision.transforms import ToTensor

class ResNet50Class(nn.Module):
    def __init__(self, class_num = 2):
        super(ResNet50Class, self).__init__()
        self.resnet50 = models.resnet50(pretrained = True)
        self.fc = nn.Sequential(nn.Linear(self.resnet50.fc.out_features, 100),
                                           nn.ReLU(),
                                           nn.Linear(100, class_num),
                                           nn.Softmax(dim=1))
    def forward(self, x):
        x = self.resnet50(x)
        x = self.fc(x)
        return x
    
class MammographyModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # ResNet50 
        self.rnet = models.resnet50(pretrained = True)
        self.rnet.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.rnet.fc = torch.nn.Linear(in_features=2048, out_features=500)
        
        # Final classification network
        self.fc1 = nn.Linear(508, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, img, meta_features):
        
        # ResNet50
        resnet_out = self.rnet(img)
        resnet_out = torch.sigmoid(resnet_out)
        
        # Reshape meta features
        meta_features = meta_features.squeeze(1)
        
        # Get final predictions
        x_final = torch.cat((resnet_out, meta_features), dim=1).to(torch.float32)        
        x_final = self.fc1(x_final)
        
        out = self.sigmoid(x_final)
        
        return out 