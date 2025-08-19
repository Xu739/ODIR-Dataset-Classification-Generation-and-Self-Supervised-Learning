import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConv2d(nn.Module):
    """基本的卷积模块：Conv2d + BatchNorm + ReLU"""

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class Inception(nn.Module):
    """Inception模块"""

    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()

        # 1x1卷积分支
        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)

        # 1x1卷积 + 3x3卷积分支
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )

        # 1x1卷积 + 5x5卷积分支
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),
            BasicConv2d(ch5x5red, ch5x5, kernel_size=5, padding=2)
        )

        # 3x3池化 + 1x1卷积分支
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):
    """辅助分类器"""

    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.conv = BasicConv2d(in_channels, 128, kernel_size=1)

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # 4x4x128
        x = self.avg_pool(x)
        x = self.conv(x)

        # 2048
        x = torch.flatten(x, 1)

        # 1024
        x = F.dropout(x, 0.5, training=self.training)
        x = F.relu(self.fc1(x), inplace=True)

        # num_classes
        x = F.dropout(x, 0.5, training=self.training)
        x = self.fc2(x)

        return x


class GoogleNet(nn.Module):
    """完整的GoogleNet模型"""

    def __init__(self, num_classes=1000, aux_logits=True):
        super(GoogleNet, self).__init__()
        self.aux_logits = aux_logits

        # 初始卷积层
        self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.conv2 = BasicConv2d(64, 64, kernel_size=1)
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        # Inception模块组
        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        # 平均池化和分类器
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)

        # 辅助分类器
        if aux_logits:
            self.aux1 = InceptionAux(512, num_classes)
            self.aux2 = InceptionAux(528, num_classes)
        else:
            self.aux1 = None
            self.aux2 = None

    def forward(self, x):
        # 输入: 3x224x224

        # 64x112x112
        x = self.conv1(x)
        x = self.maxpool1(x)

        # 64x56x56
        x = self.conv2(x)

        # 192x56x56
        x = self.conv3(x)
        x = self.maxpool2(x)

        # 256x28x28
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        # 480x14x14
        x = self.inception4a(x)

        # 辅助分类器1
        aux1 = None
        if self.aux1 is not None and self.training:
            aux1 = self.aux1(x)

        # 512x14x14
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)

        # 辅助分类器2
        aux2 = None
        if self.aux2 is not None and self.training:
            aux2 = self.aux2(x)

        # 528x14x14
        x = self.inception4e(x)
        x = self.maxpool4(x)

        # 832x7x7
        x = self.inception5a(x)
        x = self.inception5b(x)

        # 1024x7x7
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        if self.training and self.aux_logits:
            return x, aux1, aux2
        return x


