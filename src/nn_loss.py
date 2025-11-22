import torch
from torch import nn
from torch.nn import CrossEntropyLoss

input = torch.tensor([1, 2, 3], dtype=torch.float)
target = torch.tensor([1, 2, 5], dtype=torch.float)

loss = nn.L1Loss(reduction="sum")
loss_result = loss(input, target)
print(loss_result)

loss_mse = nn.MSELoss()
result_mse = loss_mse(input, target)
print(result_mse)

x = torch.tensor([[3.7, 5.0, 8.8],
                  [1.0, 9.3, 0.6]])  # logits,两个batch
y = torch.tensor([2, 1])            # target，值是真实标签0-based index
loss_cross = nn.CrossEntropyLoss()
result_cross = loss_cross(x, y)
print(result_cross)