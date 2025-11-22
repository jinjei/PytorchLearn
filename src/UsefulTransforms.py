from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter("logs")
img = Image.open("images/wallpaper.png")
print(img)

#ToTensor: [H,W,C] -> [C,H,W]，并把像素从 [0,255] 归一化到 [0,1]
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image("ToTensor", img_tensor)

#Normalize: 对每个通道做 (x - mean) / std
print(img_tensor[0][0][0])

mean = [0.668, 0.688, 0.637]
std  = [0.229, 0.224, 0.225]
norm = transforms.Normalize(mean, std)
img_norm = norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image("Normalize", img_norm)

#Resize
print(img.size)
trans_size = transforms.Resize((1000, 600))
img_resize = trans_size(img_tensor)
writer.add_image("Resize", img_resize, 0)

#Compose(Resize 2)
trans_resize_2 = transforms.Resize((600))
trans_compose = transforms.Compose([
    trans_resize_2,
    trans_totensor
])
imf_resize_2 = trans_compose(img)
writer.add_image("Resize", imf_resize_2, 1)

#RandomCrop
trans_random = transforms.RandomCrop((500, 1000))
for i in range(10):
    img_crop = trans_random(img_tensor)
    writer.add_image("RandomCrop", img_crop, i)

writer.close()