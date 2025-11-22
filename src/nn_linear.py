import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

dataset = torchvision.datasets.CIFAR10(root="../data", train=False,
                                       download=True, transform=ToTensor())
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True)


class JjxNet(nn.Module):
    def __init__(self):
        super(JjxNet, self).__init__()
        self.linear1 = nn.Linear(196608, 10)

    def forward(self, x):
        x = self.linear1(x)
        return x

jjxnet = JjxNet()

for data in dataloader:
    imgs, labels = data
    print(imgs.shape)
    output = torch.flatten(imgs) # torch.reshape(input, (1, 1, 1, -1))也可以
    print(output.shape)
    output = jjxnet(output)
    print(output.shape)