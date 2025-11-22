import torchvision
from torch import nn
from torchvision.models import VGG16_Weights

vgg16_true = torchvision.models.vgg16(pretrained=True)
vgg16_false = torchvision.models.vgg16(pretrained=False)

print(vgg16_true)

train_data = torchvision.datasets.CIFAR10("../data", train=True,
                                          download=True, transform=torchvision.transforms.ToTensor())

vgg16_true.classifier.add_module("add_linear", nn.Linear(1000, 10))
print(vgg16_true)

vgg16_false.classifier[6] = nn.Linear(1000, 10)
print(vgg16_false)