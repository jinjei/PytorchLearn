import torch

outputs = torch.tensor([[0.1, 0.3],
                        [0.2, 0.6]])

print(outputs.argmax(dim=1)) # dim=1：横着；=0纵着
preds = outputs.argmax(dim=1)

targets = torch.tensor([[0, 1]])
print((preds == targets).sum().item())