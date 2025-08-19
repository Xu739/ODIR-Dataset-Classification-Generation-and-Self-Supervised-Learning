import torch
import torch.nn as nn

class VGG19(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGG19, self).__init__()
        self.features = self._make_layers()
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

        )
        self.fc = nn.Linear(4096, num_classes),
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = self.fc(x)
        return x

    def _make_layers(self):
        """构建VGG19的卷积层（包含16个卷积层）"""
        cfg = [
            # 通道数, 卷积层数, MaxPool?
            (64, 2, False),   # Conv1: 2层64通道
            (128, 2, False),  # Conv2: 2层128通道
            (256, 4, False), # Conv3: 4层256通道
            (512, 4, False), # Conv4: 4层512通道
            (512, 4, True),  # Conv5: 4层512通道 + 池化
        ]
        layers = []
        in_channels = 3
        for out_channels, num_convs, add_pool in cfg:
            for _ in range(num_convs):
                layers += [
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                ]
                in_channels = out_channels
            if add_pool:
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """初始化权重（Kaiming正态分布）"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def vgg19(num_classes=1000):
    """构建VGG19模型实例"""
    return VGG19(num_classes=num_classes)

