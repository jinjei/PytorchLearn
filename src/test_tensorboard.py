from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter("logs")
img_path = "data/train/bees_image/36900412_92b81831ad.jpg"
img_PIL = Image.open(img_path)
img_arr = np.array(img_PIL)
print(type(img_arr))
print(img_arr.shape)

writer.add_image("train", img_arr, 1, dataformats='HWC')


for i in range(100):
    writer.add_scalar("Y=3X", 3*i, i)

writer.close()