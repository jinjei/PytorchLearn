import torch
import torchvision
from torch import nn

vgg16 = torchvision.models.vgg16(pretrained=False)


# 保存方式1: 模型结构+模型参数
# load时必须torch.load(..., weights_only=False)，因为保存了完整模型
torch.save(vgg16, "vgg16_method1.pth")

# 保存方式2：模型参数（官方推荐）
# load时必须torch.load(..., weights_only=False或者True没区别)，因为这样只保存了权重字典
torch.save(vgg16.state_dict(), "vgg16_method2.pth")


class JjxNet(nn.Module):
    def __init__(self):
        super(JjxNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3)

    def forward(self, x):
        x = self.conv1(x)
        return x

# 陷阱
jjxnet = JjxNet()
torch.save(jjxnet, "jjxnet.pth")