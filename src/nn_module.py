import torch
from torch import nn


class JjxNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input):
        output = input + 1
        return output

jjx_net = JjxNet()
x = torch.tensor([3.0, 1, 2.1]) 
output = jjx_net(x)
print(output)