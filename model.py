from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F

# Here Squeez and exitation attention mechanism is applied to use it with the Unet model

# Code for Squeeze-and-Excitation (SE) blocks adapted from:
# ai-med/squeeze_and_excitation repository (https://github.com/ai-med/squeeze_and_excitation)

class ChannelSELayer(nn.Module): # Squeeze-and-Excitation block (Channel-wise attention)
    """
    Re-implementation of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*

    """

    def __init__(self, num_channels, reduction_ratio=2):
        """

        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSELayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """

        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output tensor
        """
        batch_size, num_channels, H, W = input_tensor.size()
        # Average along each channel
        squeeze_tensor = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        # Re-weight input tensor channels
        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1, 1))
        return output_tensor


class SpatialSELayer(nn.Module):
    """
    Spatial Squeeze-and-Excitation (spatial attention)
    Ref: Roy et al., MICCAI 2018
    """

    def __init__(self, num_channels):
        """

        :param num_channels: No of input channels
        """
        super(SpatialSELayer, self).__init__()
        self.conv = nn.Conv2d(num_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor, weights=None):
        """

        :param weights: weights for few shot learning
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output_tensor
        """
        # spatial squeeze
        batch_size, channel, a, b = input_tensor.size()

        if weights is not None:
            weights = torch.mean(weights, dim=0)
            weights = weights.view(1, channel, 1, 1)
            out = F.conv2d(input_tensor, weights)
        else:
            out = self.conv(input_tensor)
        squeeze_tensor = self.sigmoid(out)

        # spatial excitation
        # print(input_tensor.size(), squeeze_tensor.size())
        squeeze_tensor = squeeze_tensor.view(batch_size, 1, a, b)
        output_tensor = torch.mul(input_tensor, squeeze_tensor)
        #output_tensor = torch.mul(input_tensor, squeeze_tensor)
        return output_tensor


class ChannelSpatialSELayer(nn.Module): # Combined Channel + Spatial Squeeze-and-Excitation
    """
    Combined Channel + Spatial Squeeze-and-Excitation
    Ref: Roy et al., MICCAI 2018
    """

    def __init__(self, num_channels, reduction_ratio=2):
        """

        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSpatialSELayer, self).__init__()
        self.cSE = ChannelSELayer(num_channels, reduction_ratio)
        self.sSE = SpatialSELayer(num_channels)

    def forward(self, input_tensor):
        """

        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output_tensor
        """
        # Take element-wise max of channel and spatial attention outputs
        output_tensor = torch.max(self.cSE(input_tensor), self.sSE(input_tensor))
        return output_tensor


# This code is adapted from https://github.com/milesial/Pytorch-UNet/tree/master
# Modifications were made to add optional SE blocks, dropout, and custom configurations for the UNet architecture.
class DoubleConv(nn.Module):
    """(Conv => BN => LeakyReLU => Dropout) * 2 with optional SE block"""

    def __init__(self, in_channels, out_channels, drop_out , SE, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        # Two convolutional blocks
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(drop_out),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(drop_out)
        )
        # Optional SE layer if 1 apply channel wise if 2 apply channel-spatial attention
        if SE == 1:
            self.se_layer = ChannelSELayer(out_channels)
        elif SE == 2:
            self.se_layer = ChannelSpatialSELayer(out_channels)
        else:
            self.se_layer = None

    def forward(self, x):

        x = self.double_conv(x)
        if self.se_layer is not None:
            x = self.se_layer(x)
        return x


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, drop_out, SE):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, drop_out, SE)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, drop_out, SE):
        super().__init__()

        
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels, drop_out, SE)
        

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    
class UNet(nn.Module):
    """U-Net with optional SE blocks and dropout"""
    def __init__(self, n_channels, n_classes, drop_out, SE,  channels = [32, 64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Encoder path
        self.inc = (DoubleConv(n_channels, channels[0], drop_out, SE)) # 256
        self.down1 = (Down(channels[0], channels[1], drop_out, SE)) # 128
        self.down2 = (Down(channels[1], channels[2], drop_out, SE)) # 64
        self.down3 = (Down(channels[2], channels[3], drop_out, SE)) # 32
        self.down4 = (Down(channels[3], channels[4], drop_out, SE)) # 16
        # Decoder path
        self.up1 = (Up(channels[4], channels[3], drop_out, SE))
        self.up2 = (Up(channels[3], channels[2], drop_out, SE))
        self.up3 = (Up(channels[2], channels[1], drop_out, SE))
        self.up4 = (Up(channels[1], channels[0], drop_out, SE))
        # Final output
        self.outc = (OutConv(channels[0], n_classes))

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        # Output
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)
