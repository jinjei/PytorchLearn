import torch
from PIL import Image
from torch.distributed.checkpoint import load_state_dict
from torchvision import transforms
from model import JjxNet

image_path = "../images/airplane.png"
image = Image.open(image_path).convert('RGB')

transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
image = transform(image)        # [3, 32, 32]
image = image.unsqueeze(0)      # [1, 3, 32, 32] 加 batch 维度
print(image.shape)

model = JjxNet()
# gpu上训练的模型想在Mac上面小规模测一下，要 map_location=torch.device('cpu')
state_dict = torch.load("../models/JjxNet_9.pth", weights_only=True, map_location=torch.device('cpu'))
model.load_state_dict(state_dict)
print(model)

model.eval()
with torch.no_grad():
    output = model(image)

print(output)
print(output.argmax(1))