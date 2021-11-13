import torch
import cv2
import torchvision.models as models
import torch.nn as nn
import cv2
from torch.autograd import Variable
from numpy import linalg as LA
import torchvision.transforms as transforms
import torchvision.models as models

import torch

model = torch.hub.load('pytorch/vision:v0.8.0', 'resnet50', pretrained=True)
# res50 = model.features
resf = torch.nn.Sequential(*(list(model.children())[:-4]))
resmp = nn.MaxPool2d((28, 28))

class resnet_50(nn.Module):
    def __init__(self):
        super(resnet_50, self).__init__()
        self.resf = resf
        self.resmp = resmp

    def forward(self, x):
        x = self.resf(x)
        return self.resmp(x)

import numpy as np
def extract_feat(model, img_path):
    model = model.eval()
    img = cv2.imread(img_path, 1)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    transf = transforms.ToTensor()
    img_tensor = transf(img)
    img_tensor = img_tensor.float()
    img_tensor = img_tensor.resize_(1, 3, 224, 224)
    feat = model.forward(Variable(img_tensor))
    feat = feat.data.numpy()
    feat = feat.squeeze()
    norm_feat = feat / LA.norm(feat)
    # print(norm_feat.shape)
    norm_feat = [i.item() for i in norm_feat]
    return norm_feat

