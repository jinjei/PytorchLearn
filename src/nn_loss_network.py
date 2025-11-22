import torch
import torchvision
from torch import nn
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())

dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

class JjxNet(nn.Module):
    def __init__(self):
        super(JjxNet, self).__init__()

        self.model1 = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x

loss = nn.CrossEntropyLoss()
jjxnet = JjxNet()
optimizer = torch.optim.SGD(jjxnet.parameters(), lr=0.01)

for epoch in range(20):
    running_loss = 0.0
    for data in dataloader:
        imgs, labels = data
        outputs = jjxnet(imgs)
        # print(outputs)
        result_loss = loss(outputs, labels)
        # print(result_loss)
        optimizer.zero_grad()
        result_loss.backward()
        optimizer.step()
        running_loss += result_loss
    print('epoch: {}, loss: {}'.format(epoch, running_loss))