import matplotlib.pyplot as plt
import time
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

def showImage(tensorImg,title):
    unloader = transforms.ToPILImage()
    image = unloader(tensorImg)
    image.save(title+'.jpg')
    
def blur(tensorImg):
    blur2d = nn.Conv2d(in_channels=3,out_channels=3,kernel_size=5,stride=1,padding=2)
    blurImg = blur2d(tensorImg)
    return blurImg