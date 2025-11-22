import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor

dataset = torchvision.datasets.CIFAR10(root="../data", train=False, download=True, transform=ToTensor())
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

class Jjxnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(x)
        return x


jjxnet = Jjxnet()
writer = SummaryWriter("../logs_sigmoid")
step = 0
for data in dataloader:
    imgs, labels = data
    writer.add_images("input", imgs, step)
    output = jjxnet(imgs)
    writer.add_images("output", output, step)
    step += 1

writer.close()