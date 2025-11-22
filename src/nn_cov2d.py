import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

class JjxNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        return x


jjxnet = JjxNet()
print(jjxnet)

writer = SummaryWriter("logs")
step = 0
for data in dataloader:
    imgs, targets = data
    output = jjxnet(imgs)
    print(imgs.shape)
    print(output.shape)

    # torch.size([64, 3, 32, 32])
    writer.add_images("input", imgs, step)
    # torch.size([64, 6, 32, 32]) -> torch.size([64, 3, 32, 32])
    output = torch.reshape(output, (-1, 3, 32, 32))
    writer.add_images("output", output, step)

    step += 1