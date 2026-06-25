__all__ = (
    "MAP",
    "DMCABlock",
    "MaxPool",
    "AvgPool",
    "Concat_ZFQ",
    "C2f_ZFQ",
    "MSGConv",
    "Lw_C2f_ZFQ",
    "ECALayer",

)

import math
from . import Bottleneck
from .conv import Conv, DWConv

import torch
import torch.nn as nn


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class CA_Block(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CA_Block, self).__init__()

        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel // reduction, kernel_size=1, stride=1,
                                  bias=False)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel // reduction)

        self.F_h = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)
        self.F_w = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        _, _, h, w = x.size()

        x_h = torch.mean(x, dim=3, keepdim=True).permute(0, 1, 3, 2)
        x_w = torch.mean(x, dim=2, keepdim=True)

        x_cat_conv_relu = self.relu(self.bn(self.conv_1x1(torch.cat((x_h, x_w), 3))))

        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([h, w], 3)

        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))

        out = x * s_h.expand_as(x) * s_w.expand_as(x)
        return out


class ECALayer(nn.Module):
    """
    《ECANet：Efficient Channel Attention for Deep Convolutional Neural Networks》
    高效通道注意力机制
    Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
    """

    def __init__(self, channel, b=1, gamma=2):
        super(ECALayer, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class CPSPPSELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        """《Spatial Pyramid Attention Network for Enhanced Image Recognition》
        空间金字塔注意力机制
        """
        super(CPSPPSELayer, self).__init__()
        self.avg_pool1 = nn.AdaptiveAvgPool2d(1)
        self.avg_pool2 = nn.AdaptiveAvgPool2d(2)
        self.avg_pool3 = nn.AdaptiveAvgPool2d(4)
        self.fc = nn.Sequential(
            nn.Linear(channel * 21, channel * 21 // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel * 21 // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y1 = self.avg_pool1(x).view(b, c)  # like resize() in numpy
        y2 = self.avg_pool2(x).view(b, 4 * c)
        y3 = self.avg_pool3(x).view(b, 16 * c)
        y = torch.cat((y1, y2, y3), 1)
        y = self.fc(y)
        b, out_channel = y.size()
        y = y.view(b, out_channel, 1, 1)
        return x * y.expand_as(x)


class EPPCA(nn.Module):
    """
    高效空间金字塔串行池化通道注意力机制
    Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
    """

    def __init__(self, channel, b=1, gamma=2):
        """《Spatial Pyramid Attention Network for Enhanced Image Recognition》
        空间金字塔注意力机制
        """
        super(EPPCA, self).__init__()

        self.avg_pool1 = nn.AdaptiveAvgPool2d(4)
        self.avg_pool2 = nn.AdaptiveAvgPool2d(2)
        self.avg_pool3 = nn.AdaptiveAvgPool2d(1)

        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.conv1 = nn.Conv1d(channel, channel, 21, groups=channel)
        self.conv2 = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        out1 = self.avg_pool1(x)
        out2 = self.avg_pool2(out1)
        out3 = self.avg_pool3(out2)
        y1 = out1.view(b, c, 1, 16)  # like resize() in numpy
        y2 = out2.view(b, c, 1, 4)
        y3 = out3.view(b, c, 1, 1)
        y = torch.cat((y1, y2, y3), 3)
        y = self.conv1(y.view(b, c, 21))
        y = self.conv2(y.view(b, 1, c))
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Concat_ZFQ(nn.Module):
    """Concatenate a list of tensors along dimension."""

    def __init__(self, c, dimension=1):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        self.c = None
        self.d = dimension
        self.CA = EPPCA(c)

    def forward(self, x):
        """Forward pass for the YOLOv8 mask Proto module."""
        y = torch.cat(x, self.d)
        return self.CA(y)


class MaxPool(nn.Module):
    """并行最大、均值池化-通道拼接扩充通道维度"""

    def __init__(self, c1, c2, k=3, s=2):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=k, stride=s, padding=1)
        self.cv = Conv(c1, c2, 1, 1)

    def forward(self, x):
        y = self.cv(self.pool(x))
        return y


class AvgPool(nn.Module):
    """并行最大、均值池化-通道拼接扩充通道维度"""

    def __init__(self, c1, c2, k=3, s=2):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=k, stride=s, padding=1)
        self.cv = Conv(c1, c2, 1, 1)

    def forward(self, x):
        y = self.cv(self.pool(x))
        return y


class MAP(nn.Module):
    """并行最大、均值池化-通道拼接扩充通道维度"""

    def __init__(self, c1, c2, k=3, s=2):
        super().__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=k, stride=s, padding=1)
        self.pool2 = nn.AvgPool2d(kernel_size=k, stride=s, padding=1)

        self.CA = nn.Sequential(
            ECALayer(c1 * 2),
            nn.Conv2d(c1 * 2, c2, (1, 1), 1),
        )

    def forward(self, x):
        y = list(self.pool1(x).chunk(1, 1))
        y.extend(self.pool2(x).chunk(1, 1))
        y = torch.cat(y, 1)
        y = self.CA(y)
        return y


class DMCABlock(nn.Module):
    def __init__(self, channels, b=1, gamma=2):
        super(DMCABlock, self).__init__()

        # =====================================================
        # 1)多尺度最大 / 均值池化（逐级相加）
        # =====================================================
        ks = [3, 5, 7]
        self.max_pools = nn.ModuleList([nn.MaxPool2d(k, 1, k // 2) for k in ks])
        self.avg_pools = nn.ModuleList([nn.AvgPool2d(k, 1, k // 2) for k in ks])

        # =====================================================
        # 3) 坐标注意力 CA（取消 reduce）
        # =====================================================
        self.cbr = nn.Sequential(
            nn.Conv2d(channels, channels, 1, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        self.F_h = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1, stride=1,
                             bias=False)
        self.F_w = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1, stride=1,
                             bias=False)

        # =====================================================
        # 4) 通道注意力 ECA（保持不变）
        # =====================================================
        kernel_size = int(abs((math.log(channels, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.conv_channel = nn.Conv1d(
            channels, channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=False
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()

        # =====================================================
        # 1) 多尺度池化
        # =====================================================
        # 图中的三次加法结构
        xav1 = self.avg_pools[0](x)
        xav2 = self.avg_pools[1](xav1)
        xav3 = self.avg_pools[2](xav2)
        xma1 = self.max_pools[0](x)
        xma2 = self.max_pools[1](xma1)
        xma3 = self.max_pools[2](xma2)

        feat_ms = (x + xav1 + xma1 + xav2 + xma2 + xav3 + xma3)

        # =====================================================
        # 2) 坐标注意力 CA
        # =====================================================
        x_h = torch.mean(x, dim=3, keepdim=True).permute(0, 1, 3, 2)
        x_w = torch.mean(x, dim=2, keepdim=True)

        # concat along width
        xy = torch.cat([x_h, x_w], dim=3)

        # 不降维
        ca_f = self.cbr(xy)

        # split spatially
        x_cat_conv_split_h, x_cat_conv_split_w = ca_f.split([h, w], 3)

        s_h = self.sigmoid(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid(self.F_w(x_cat_conv_split_w))

        out1 = x * s_h.expand_as(x) * s_w.expand_as(x)

        # =====================================================
        # 3) 通道注意力 ECA
        # =====================================================
        gap = nn.functional.adaptive_avg_pool2d(feat_ms, 1).squeeze(-1)  # [B,C,1]
        A_c = self.sigmoid(self.conv_channel(gap).unsqueeze(-1))  # [B,C,1,1]

        # =====================================================
        # 4) 输出
        # =====================================================
        return out1 + x * A_c


class MSGConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        c = out_channels // 4

        self.conv3 = nn.Conv2d(c, c, 3, 1, 1, bias=False)
        self.dw5 = nn.Sequential(
            nn.Conv2d(c, c, 5, 1, 2, groups=c, bias=False),
            nn.Conv2d(c, c, 1)
        )
        self.dw7 = nn.Sequential(
            nn.Conv2d(c, c, 7, 1, 3, groups=c, bias=False),
            nn.Conv2d(c, c, 1)
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)

        y1 = self.conv3(x2)
        y2 = self.dw5(x3)
        y3 = self.dw7(x4)

        y = torch.cat([x1, y1, y2, y3], dim=1)
        return self.act(self.bn(y))


class Bottleneck_MSGC(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = MSGConv(c1, c_)
        self.cv2 = MSGConv(c_, c2)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f_ZFQ(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))
        self.CA = ECALayer((2 + n) * self.c)

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = self.cv1(x)
        y1 = y[:, 0::2, :, :]
        y2 = y[:, 1::2, :, :]
        y = [y1, y2]
        y.extend(m(y2) for m in self.m)
        y = torch.cat(y, 1)
        y = self.CA(y)
        return self.cv2(y)


class Lw_C2f_ZFQ(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            Bottleneck_MSGC(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.CA = ECALayer((2 + n) * self.c)

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = self.cv1(x)
        y1 = y[:, 0::2, :, :]
        y2 = y[:, 1::2, :, :]
        y = [y1, y2]
        y.extend(m(y2) for m in self.m)

        # y = list(self.cv1(x).chunk(2, 1))
        # y.extend(m(y[-1]) for m in self.m)

        y = torch.cat(y, 1)
        y = self.CA(y)
        return self.cv2(y)
