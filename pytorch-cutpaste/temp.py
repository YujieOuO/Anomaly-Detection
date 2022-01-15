from torchvision.models import resnet18,wide_resnet50_2
import torch
model = wide_resnet50_2(pretrained = True, progress = True)

img = torch.rand(1,3,300,300)
img2 = model(img)
print(img2.shape)