import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_model_summary import summary


class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(out_ch),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(out_ch),
                                  nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.mpconv = nn.Sequential(nn.MaxPool2d(2),
                                    double_conv(in_ch, out_ch))

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, input_channels, output_channel):
        super(UNet, self).__init__()
        self.inc = inconv(input_channels, 32)
        self.down1 = down(32, 64)
        self.down2 = down(64, 128)
        self.down3 = down(128, 256)
        self.down4 = down(256, 512)

        self.up1 = up(512, 256)
        self.up2 = up(256, 128)
        self.up3 = up(128, 64)
        self.up4 = up(64, 32)
        self.outc = nn.Conv2d(32, output_channel, kernel_size=3, padding=1)

        self.size_down8 = nn.Conv2d(1, 1, kernel_size=8, stride=8)
        self.size_down4 = nn.Conv2d(1, 1, kernel_size=4, stride=4)
        self.size_down2 = nn.Conv2d(1, 1, kernel_size=2, stride=2)

    def forward(self, x):
        input = x
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4) + self.size_down8(input)
        x = self.up2(x, x3) + self.size_down4(input)
        x = self.up3(x, x2) + self.size_down2(input)
        x = self.up4(x, x1) + input
        x = self.outc(x) 

        return x


if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    x = torch.ones([4, 1, 256, 256]).cuda()
    model = UNet(1, 1).cuda()

    print(summary(model,x))
    print('input:',x.shape)
    print('output:',model(x).shape)
    print('===================================')
    print(model.parameters)