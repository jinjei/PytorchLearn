import torchvision
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from model import *

# 准备数据集
train_data = torchvision.datasets.CIFAR10(root="../data", train=True,
                                          download=True, transform=transforms.ToTensor())
test_data = torchvision.datasets.CIFAR10(root="../data", train=False,
                                         download=True, transform=transforms.ToTensor())

train_data_size = len(train_data)
test_data_size = len(test_data)
print("Length of training set: {}".format(train_data_size))
print("Length of test set: {}".format(test_data_size))

# 利用DataLoader加载数据集
train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# 创建网络模型
jjxnet = JjxNet()

# 损失函数
loss_fn = nn.CrossEntropyLoss()

# 优化器
# learning_rate = 0.01
# 1e-2 = 1 × (10)^(-2) = 1 / 100 = 0.01
learning_rate = 1e-2
optimizer = torch.optim.SGD(jjxnet.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练轮数
epoch = 10

# 添加tensorboard
writer = SummaryWriter("../logs_train")

for i in range(epoch):
    print("------ Epoch {} Start ------".format(i+1))

    # 训练步骤开始
    jjxnet.train()
    for data in train_dataloader:
        imgs, labels = data
        outputs = jjxnet(imgs)
        loss = loss_fn(outputs, labels)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step += 1
        if total_train_step % 100 == 0:
            print("Training steps: {}, Loss: {}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤开始
    jjxnet.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, labels = data
            outputs = jjxnet(imgs)
            loss = loss_fn(outputs, labels)
            total_test_loss += loss.item()
            accuracy = (outputs.argmax(1) == labels).sum()
            total_accuracy += accuracy.item()

    print("Testing loss on overall test set: {}".format(total_test_loss))
    print("Testing accuracy on overall test set: {}".format(total_accuracy / test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy / test_data_size, total_test_step)
    total_test_step += 1

    torch.save(jjxnet.state_dict(), "../models/jjxnet_{}.pth".format(i))
    print("Model Saved")

writer.close()