import torch
import torchvision
from torch import nn
from model_save import *

# 方式1 -> 对应保存方式1   加载模型
model1 = torch.load("vgg16_method1.pth", weights_only=False)
print(model1)


# 方式2 加载模型
vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load("vgg16_method2.pth", weights_only=False))
# model2 = torch.load("vgg16_method2.pth", weights_only=False)
print(vgg16)

# 陷阱: 自己定义的网络必须写一遍类声明，不然就得import
# class JjxNet(nn.Module):
#     def __init__(self):
#         super(JjxNet, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         return x

model2 = torch.load("jjxnet.pth", weights_only=False)
print(model2)