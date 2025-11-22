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
        # super(当前类名， 当前对象)  手动告诉 super：我是谁（JjxNet），我是哪个实例（self）
        # python3直接写做：super().__init__()
        super(Jjxnet, self).__init__()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, ceil_mode=True)

    def forward(self, input):
        output = self.maxpool1(input)
        return output

writer = SummaryWriter("../logs_maxpool")
jjxnet = Jjxnet()

step = 0
for data in dataloader:
    imgs, labels = data
    writer.add_images("input", imgs, step)
    output = jjxnet(imgs)
    writer.add_images("output", output, step)
    step += 1

writer.close()
