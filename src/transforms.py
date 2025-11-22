from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image

# tesnor 数据类型
# 通过transforms.ToTensor去看两个问题

# 2. 为什么我们需要Tensor数据类型

img_path = "dataset/train/ants/6240338_93729615ec.jpg"
img = Image.open(img_path)

writer = SummaryWriter("logs")
# 1. transforms如何使用（python）
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)

writer.add_image("Tensor_img", tensor_img)
writer.close()