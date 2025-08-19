import torch
import torch.nn as nn
import math
from typing import List, Callable

# 激活函数（默认使用 Swish）
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# 深度可分离卷积（Depthwise Separable Convolution）
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = Swish()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.act(x)
        return x

# MBConv 块（EfficientNet 核心模块）
class MBConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        expansion_factor: int,
        se_ratio: float = 0.25,
    ):
        super().__init__()
        expanded_channels = in_channels * expansion_factor
        self.use_residual = (in_channels == out_channels) and (stride == 1)

        # Expansion phase (1x1 conv)
        if expansion_factor != 1:
            self.expand = nn.Sequential(
                nn.Conv2d(in_channels, expanded_channels, 1, bias=False),
                nn.BatchNorm2d(expanded_channels),
                Swish(),
            )
        else:
            self.expand = nn.Identity()

        # Depthwise convolution
        self.depthwise = nn.Sequential(
            nn.Conv2d(
                expanded_channels,
                expanded_channels,
                kernel_size,
                stride,
                kernel_size // 2,
                groups=expanded_channels,
                bias=False,
            ),
            nn.BatchNorm2d(expanded_channels),
            Swish(),
        )

        # Squeeze-and-Excitation (SE) block
        squeeze_channels = max(1, int(in_channels * se_ratio))
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(expanded_channels, squeeze_channels, 1),
            Swish(),
            nn.Conv2d(squeeze_channels, expanded_channels, 1),
            nn.Sigmoid(),
        )

        # Pointwise convolution (1x1)
        self.pointwise = nn.Sequential(
            nn.Conv2d(expanded_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        residual = x
        x = self.expand(x)
        x = self.depthwise(x)
        x = self.se(x) * x  # SE block
        x = self.pointwise(x)

        if self.use_residual:
            x += residual

        return x

# EfficientNet 主模型
class EfficientNet(nn.Module):
    def __init__(
        self,
        width_mult: float = 1.0,
        depth_mult: float = 1.0,
        dropout_rate: float = 0.2,
        num_classes: int = 1000,
    ):
        super().__init__()
        channels = [32, 16, 24, 40, 80, 112, 192, 320, 1280]
        repeats = [1, 2, 2, 3, 3, 4, 1]
        strides = [1, 2, 2, 2, 1, 2, 1]
        kernel_sizes = [3, 3, 5, 3, 5, 5, 3]
        expansion_factors = [1, 6, 6, 6, 6, 6, 6]

        # 调整宽度和深度
        channels = [int(c * width_mult) for c in channels]
        repeats = [int(r * depth_mult) for r in repeats]

        # 初始卷积层
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels[0], 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            Swish(),
        )

        # MBConv 块堆叠
        blocks = []
        for i in range(7):
            in_c = channels[i]
            out_c = channels[i + 1]
            for j in range(repeats[i]):
                stride = strides[i] if j == 0 else 1
                blocks.append(
                    MBConv(
                        in_c if j == 0 else out_c,
                        out_c,
                        kernel_sizes[i],
                        stride,
                        expansion_factors[i],
                    )
                )
        self.blocks = nn.Sequential(*blocks)

        # 分类层
        self.head = nn.Sequential(
            nn.Conv2d(channels[-2], channels[-1], 1, bias=False),
            nn.BatchNorm2d(channels[-1]),
            Swish(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout_rate),

        )
        self.fc = nn.Linear(channels[-1], num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        x = self.fc(x)
        return x

# EfficientNet-B0 到 B7 的配置
def efficientnet_b0(num_classes=1000):
    return EfficientNet(1.0, 1.0, 0.2, num_classes)

def efficientnet_b1(num_classes=1000):
    return EfficientNet(1.0, 1.1, 0.2, num_classes)

def efficientnet_b2(num_classes=1000):
    return EfficientNet(1.1, 1.2, 0.3, num_classes)

def efficientnet_b3(num_classes=1000):
    return EfficientNet(1.2, 1.4, 0.3, num_classes)

def efficientnet_b4(num_classes=1000):
    return EfficientNet(1.4, 1.8, 0.4, num_classes)

def efficientnet_b5(num_classes=1000):
    return EfficientNet(1.6, 2.2, 0.4, num_classes)

def efficientnet_b6(num_classes=1000):
    return EfficientNet(1.8, 2.6, 0.5, num_classes)

def efficientnet_b7(num_classes=1000):
    return EfficientNet(2.0, 3.1, 0.5, num_classes)

# 测试代码
if __name__ == "__main__":
    model = efficientnet_b0(num_classes=10)
    input_tensor = torch.randn(1, 3, 224, 224)  # 输入尺寸需匹配（B0 默认 224x224）
    output = model(input_tensor)
    print(f"Output shape: {output.shape}")  # 应该输出: torch.Size([1, 10])