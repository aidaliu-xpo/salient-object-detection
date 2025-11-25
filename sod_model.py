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
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        #kerneli 3x3

    def forward(self, x):
        #apply convolution (sliding the kernel through the image to form the feature map)
        x = self.conv(x)

        #keep positive values, zero out the negative ones
        x = F.relu(x)
        return x


#UbBlock takes a small feature map and upsamples it back up
class UpBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        #ConvTranspose2d up samples the feature map
        #stride=2 -> doubles height and width (reverse pooling)
        self.up = nn.ConvTranspose2d(
            in_channels, out_channels,
            kernel_size=2, stride=2
        )

    def forward(self, x):

        #ConvTransposed2d performs learned upsampling: it increases the spatial size
        #by using trainable filters. The model learns how to recreate details when
        #expanding the feature map (not just stretching pixels).
        x = self.up(x)

        x = F.relu(x)

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

        #Decoder
        self.dec1 = UpBlock(64, 32)
        self.dec2 = UpBlock(32, 16)
        self.dec3 = UpBlock(16, 8)

        #Final output layer
        self.final_conv = nn.Conv2d(8, 1, kernel_size=1)


    def forward(self, x):

        #Encoder
        x = self.enc1(x)
        x = self.pool1(x)

        x = self.enc2(x)
        x = self.pool2(x)

        x = self.enc3(x)
        x = self.pool3(x)

        #Decoder
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)

        #Final Mask
        x = self.final_conv(x)
        x = torch.sigmoid(x)

        return x
