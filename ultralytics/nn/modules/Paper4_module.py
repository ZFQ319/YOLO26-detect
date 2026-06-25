__all__ = (
    "SGAM",
    "StrC2f"
)

from ultralytics.nn.modules import Conv, C2f
from ultralytics.nn.modules.my_module import ECALayer, CA_Block

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, H, W]

        avg_out = torch.mean(x, dim=1, keepdim=True)  # GAP
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # GMP

        x = torch.cat([avg_out, max_out], dim=1)  # [B, 2, H, W]
        x = self.conv(x)
        return self.sigmoid(x)  # [B,1,H,W]


class SGAM(nn.Module):
    def __init__(self, c_high, c_low):
        super().__init__()

        # 高层语义 → 空间注意力
        self.spatial_att = SpatialAttention()
        # 通道注意力
        self.CA = CA_Block(c_high)

    def forward(self, x):
        f_high, f_low = x

        # 上采样对齐
        f_high = F.interpolate(f_high, size=f_low.shape[-2:], mode='nearest')

        # CA注意力
        f_high =self.CA(f_high) + f_high

        # 空间注意力（语义引导）
        att = self.spatial_att(f_high)

        # 高维特征语义引导注意力增强
        f_low_enhanced = f_low * (1 + att)

        # 高层参与融合
        f_out = torch.cat([f_high, f_low_enhanced], dim=1)

        return f_out


class StripConvBottleneck(nn.Module):
    def __init__(
            self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k = (3, 3), e: float = 0.5
    ):
        """Initialize a standard bottleneck module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            g (int): Groups for convolutions.
            k (tuple[int, int] | tuple[tuple[int, int],...): Kernel sizes for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, (5,1), 1, g=g)
        self.cv2 = Conv(c_, c_, (1,5), 1, g=g)
        self.cv3 = Conv(c_, c2, (3,3), 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply bottleneck with optional shortcut connection."""
        return x + self.cv3(self.cv2(self.cv1(x))) if self.add else self.cv2(self.cv1(x))


class StrC2f(nn.Module):

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):

        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(StripConvBottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
