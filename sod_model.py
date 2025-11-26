import torch
import torch.nn as nn
import torch.nn.functional as F


# this is the actual convolution layer
class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        #padding adds fake pixels at the border of the image, so the kernel has room to slide even on the edges
        #in channels -- how many feature maps come in
        #out channels -- how many feature maps go out
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, padding=1)

        self.bn2 = nn.BatchNorm2d(out_channels)

        #kerneli 3x3

    def forward(self, x):
        #apply convolution (sliding the kernel through the image to form the feature map)
        #keep positive values, zero out the negative ones
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


#UbBlock takes a small feature map and upsamples it back up
class UpBlock(nn.Module):

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        #ConvTranspose2d up samples the feature map
        #stride=2 -> doubles height and width (reverse pooling)
        self.up = nn.ConvTranspose2d(
            in_channels, out_channels,
            kernel_size=2, stride=2
        )

        self.conv = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):

        #ConvTransposed2d performs learned upsampling: it increases the spatial size
        #by using trainable filters. The model learns how to recreate details when
        #expanding the feature map (not just stretching pixels).
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)

        return x


class SaliencyNet(nn.Module):

    def __init__(self):
        super().__init__()

        #Encoder
        self.enc1 = ConvBlock(3, 16)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(16, 32)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(32, 64)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(64, 128)

        #Decoder
        self.dec1 = UpBlock(in_channels=128, skip_channels=64, out_channels=64)
        self.dec2 = UpBlock(in_channels=64, skip_channels=32, out_channels=32)
        self.dec3 = UpBlock(in_channels=32, skip_channels=16, out_channels=16)

        #Final output layer
        self.final_conv = nn.Conv2d(16, 1, kernel_size=1)


    def forward(self, x):

        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        b = self.bottleneck(p3)

        d1 = self.dec1(b, e3)
        d2 = self.dec2(d1, e2)
        d3 = self.dec3(d2, e1)

        x = self.final_conv(d3)
        x = torch.sigmoid(x)

        return x
