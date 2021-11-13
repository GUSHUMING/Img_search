import torchvision.models as models
import torch.nn as nn
import cv2
from torch.autograd import Variable
from numpy import linalg as LA
import torchvision.transforms as transforms

vgg16_pre = models.vgg16(pretrained=True)
vggf = vgg16_pre.features
vggmp = nn.MaxPool2d((7, 7))


class vgg16_net(nn.Module):
    def __init__(self):
        super(vgg16_net, self).__init__()
        self.vggf = vggf
        self.vggmp = vggmp

    def forward(self, x):
        x = self.vggf(x)
        return self.vggmp(x)


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

