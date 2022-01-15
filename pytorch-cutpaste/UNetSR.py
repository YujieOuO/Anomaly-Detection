"""
@author: Huakun Shen, Abhishek Thakur
@reference: https://youtu.be/u1loyDCoGbE
"""
import torch
from torch import nn
from PIL import Image
from torch.nn.modules import padding
from torchvision import transforms
import torch.nn.functional as F


def double_convolution(in_c, out_c, ksize=3):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=ksize, padding=ksize//2),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=ksize, padding=ksize//2),
        nn.ReLU(inplace=True),
    )


class UNetSR(nn.Module):
    """
    unetsr = UNetSR(in_c=3, out_c=3, output_paddings=[1, 1]).to(device)
    unet_config = {
        'epochs': 150,
        'save_period': 10,
        'batch_size': 8,
        'checkpoint_dir': RESULT_PATH / 'result/unetsr_100_300_perceptual_loss_w_seed',
        'log_step': 10,
        'start_epoch': 1,
        'criterion': criterion,
        'dataset_type': 'same_300',
        'low_res': 100,
        'high_res': 300,
        'device': device,
        'scheduler': {
            'step_size': 5,
            'gamma': 0.85
        },
        'optimizer': optim.Adam(unetsr.parameters(), lr=0.002),
        'train_set_percentage': 0.9,
        'num_worker': multiprocessing.cpu_count(),
        'test_all_multiprocess_cpu': 1,
        'test_only': False
    }

    """

    def __init__(self, in_c: int = 3, out_c: int = 3, ksize=3, output_paddings=[1, 1]):
        """output_paddings: second number is 0 when input size is 600, 1 if input size is 300"""
        super(UNetSR, self).__init__()
        self.MaxPool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        # encoder part
        self.encoder_conv_1 = double_convolution(in_c, 64, ksize=ksize)
        self.encoder_conv_2 = double_convolution(64, 128, ksize=ksize)
        self.encoder_conv_3 = double_convolution(128, 256, ksize=ksize)
        self.encoder_conv_4 = double_convolution(256, 512, ksize=ksize)
        self.encoder_conv_5 = double_convolution(512, 1024, ksize=ksize)
        # decoder part
        self.ConvT2D_1 = nn.ConvTranspose2d(
            in_channels=1024, out_channels=512, kernel_size=2, stride=2, output_padding=output_paddings[0])
        self.decoder_conv_1 = double_convolution(1024, 512, ksize=ksize)
        self.ConvT2D_2 = nn.ConvTranspose2d(
            in_channels=512, out_channels=256, kernel_size=2, stride=2, output_padding=output_paddings[1])
        self.decoder_conv_2 = double_convolution(512, 256, ksize=ksize)
        self.ConvT2D_3 = nn.ConvTranspose2d(
            in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.decoder_conv_3 = double_convolution(256, 128, ksize=ksize)
        self.ConvT2D_4 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.decoder_conv_4 = double_convolution(128, 64, ksize=ksize)
        # output layer to 3 channels
        self.final = nn.Conv2d(64, out_c, kernel_size=1)

    def forward(self, image):
        x1 = self.encoder_conv_1(image)    # to be concatenated to decoder
        x2 = self.MaxPool2d(x1)
        # print(1, x2.shape)

        x3 = self.encoder_conv_2(x2)       # to be concatenated to decoder
        x4 = self.MaxPool2d(x3)
        # print(2, x4.shape)

        x5 = self.encoder_conv_3(x4)       # to be concatenated to decoder
        x6 = self.MaxPool2d(x5)
        # print(3, x6.shape)

        x7 = self.encoder_conv_4(x6)       # to be concatenated to decoder
        x8 = self.MaxPool2d(x7)
        # print(4, x8.shape)

        x9 = self.encoder_conv_5(x8)
        # print(5, x9.shape)

        x = self.ConvT2D_1(x9)
        # print(6, x.shape)

        x = self.decoder_conv_1(torch.cat([x, x7], 1))
        # print(7, x.shape)

        x = self.ConvT2D_2(x)
        # print(8, x.shape)

        x = self.decoder_conv_2(torch.cat([x, x5], 1))
        # print(9, x.shape)

        x = self.ConvT2D_3(x)
        x = self.decoder_conv_3(torch.cat([x, x3], 1))
        # print(10, x.shape)

        x = self.ConvT2D_4(x)
        # print(11, x.shape)

        x = self.decoder_conv_4(torch.cat([x, x1], 1))
        # print(12, x.shape)

        x = self.final(x)
        # print(13, x.shape)
        return x


class UNetNoTop(nn.Module):
    """
    remove top layer skip connection
    """

    def __init__(self, in_c: int = 3, out_c: int = 3, ksize=3):
        super(UNetNoTop, self).__init__()
        self.MaxPool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        # encoder part
        self.encoder_conv_1 = double_convolution(in_c, 64, ksize=ksize)
        self.encoder_conv_2 = double_convolution(64, 128, ksize=ksize)
        self.encoder_conv_3 = double_convolution(128, 256, ksize=ksize)
        self.encoder_conv_4 = double_convolution(256, 512, ksize=ksize)
        self.encoder_conv_5 = double_convolution(512, 1024, ksize=ksize)
        # decoder part
        self.ConvT2D_1 = nn.ConvTranspose2d(
            in_channels=1024, out_channels=512, kernel_size=2, stride=2, output_padding=1)
        self.decoder_conv_1 = double_convolution(1024, 512, ksize=ksize)
        self.ConvT2D_2 = nn.ConvTranspose2d(
            in_channels=512, out_channels=256, kernel_size=2, stride=2, output_padding=1)
        self.decoder_conv_2 = double_convolution(512, 256, ksize=ksize)
        self.ConvT2D_3 = nn.ConvTranspose2d(
            in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.decoder_conv_3 = double_convolution(256, 128, ksize=ksize)
        self.ConvT2D_4 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.decoder_conv_4 = double_convolution(64, 64, ksize=ksize)
        # output layer to 3 channels
        self.final = nn.Conv2d(64, out_c, kernel_size=1)

    def forward(self, image):
        x1 = self.encoder_conv_1(image)    # to be concatenated to decoder
        x2 = self.MaxPool2d(x1)
        x3 = self.encoder_conv_2(x2)       # to be concatenated to decoder
        x4 = self.MaxPool2d(x3)
        x5 = self.encoder_conv_3(x4)       # to be concatenated to decoder
        x6 = self.MaxPool2d(x5)
        x7 = self.encoder_conv_4(x6)       # to be concatenated to decoder
        x8 = self.MaxPool2d(x7)
        x9 = self.encoder_conv_5(x8)
        x = self.ConvT2D_1(x9)
        x = self.decoder_conv_1(torch.cat([x, x7], 1))
        x = self.ConvT2D_2(x)
        x = self.decoder_conv_2(torch.cat([x, x5], 1))
        x = self.ConvT2D_3(x)
        x = self.decoder_conv_3(torch.cat([x, x3], 1))
        x = self.ConvT2D_4(x)
        # x = self.decoder_conv_4(torch.cat([x, x1], 1))
        x = self.decoder_conv_4(x)
        x = self.final(x)
        return x


class UNetD4(nn.Module):
    """
    UNet depth=4, instead of original 5 layers
    """

    def __init__(self, in_c: int = 3, out_c: int = 3, ksize=3):
        super(UNetD4, self).__init__()
        self.MaxPool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        # encoder part
        self.encoder_conv_1 = double_convolution(in_c, 64, ksize=ksize)
        self.encoder_conv_2 = double_convolution(64, 128, ksize=ksize)
        self.encoder_conv_3 = double_convolution(128, 256, ksize=ksize)
        self.encoder_conv_4 = double_convolution(256, 512, ksize=ksize)
        self.encoder_conv_5 = double_convolution(512, 1024, ksize=ksize)
        # decoder part
        self.ConvT2D_1 = nn.ConvTranspose2d(
            in_channels=1024, out_channels=512, kernel_size=2, stride=2, output_padding=1)
        self.decoder_conv_1 = double_convolution(1024, 512, ksize=ksize)
        self.ConvT2D_2 = nn.ConvTranspose2d(
            in_channels=512, out_channels=256, kernel_size=2, stride=2, output_padding=1)
        self.decoder_conv_2 = double_convolution(512, 256, ksize=ksize)
        self.ConvT2D_3 = nn.ConvTranspose2d(
            in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.decoder_conv_3 = double_convolution(256, 128, ksize=ksize)
        self.ConvT2D_4 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.decoder_conv_4 = double_convolution(128, 64, ksize=ksize)
        # output layer to 3 channels
        self.final = nn.Conv2d(64, out_c, kernel_size=1)

    def forward(self, image):
        x1 = self.encoder_conv_1(image)    # to be concatenated to decoder
        x2 = self.MaxPool2d(x1)

        x3 = self.encoder_conv_2(x2)       # to be concatenated to decoder
        x4 = self.MaxPool2d(x3)

        x5 = self.encoder_conv_3(x4)       # to be concatenated to decoder
        x6 = self.MaxPool2d(x5)

        x7 = self.encoder_conv_4(x6)       # to be concatenated to decoder
        x = self.ConvT2D_2(x7)

        x = self.decoder_conv_2(torch.cat([x, x5], 1))

        x = self.ConvT2D_3(x)
        x = self.decoder_conv_3(torch.cat([x, x3], 1))

        x = self.ConvT2D_4(x)

        x = self.decoder_conv_4(torch.cat([x, x1], 1))

        x = self.final(x)
        return x


if __name__ == "__main__":
    image = torch.rand((64, 3, 300, 300))
    print(image.size())
    #model = UNetSR()
    # model = UNetNoTop()
    model = UNetD4()
    # model = UNetSR(output_paddings=[1, 0])
    out = model(image)
    print(out.shape)
