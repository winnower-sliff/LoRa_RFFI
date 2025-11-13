# net/net_mobilenet.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积层"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, 
                                   stride=stride, padding=1, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class MobileNet(nn.Module):
    """MobileNet 特征提取器"""
    def __init__(self, in_channels, width_multiplier=1.0):
        super(MobileNet, self).__init__()
        self.width_multiplier = width_multiplier
        
        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )
        
        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
                
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )
        
        # 根据宽度乘数调整通道数
        output_channels = int(32 * width_multiplier)
        self.model = nn.Sequential(
            conv_bn(in_channels, output_channels, 2),
            conv_dw(output_channels, int(64*width_multiplier), 1),
            conv_dw(int(64*width_multiplier), int(128*width_multiplier), 2),
            conv_dw(int(128*width_multiplier), int(128*width_multiplier), 1),
            conv_dw(int(128*width_multiplier), int(256*width_multiplier), 2),
            conv_dw(int(256*width_multiplier), int(256*width_multiplier), 1),
            conv_dw(int(256*width_multiplier), int(512*width_multiplier), 2),
            conv_dw(int(512*width_multiplier), int(512*width_multiplier), 1),
            conv_dw(int(512*width_multiplier), int(512*width_multiplier), 1),
            conv_dw(int(512*width_multiplier), int(512*width_multiplier), 1),
            conv_dw(int(512*width_multiplier), int(512*width_multiplier), 1),
            conv_dw(int(512*width_multiplier), int(512*width_multiplier), 1),
            conv_dw(int(512*width_multiplier), int(1024*width_multiplier), 2),
            conv_dw(int(1024*width_multiplier), int(1024*width_multiplier), 1),
        )
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(int(1024*width_multiplier), 512)

    def forward(self, x):
        x = self.model(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = F.normalize(self.fc(x), p=2, dim=1)  # L2 正则化
        return x


def mobilenet(in_channels, width_multiplier=1.0):
    """返回 MobileNet 对象"""
    return MobileNet(in_channels, width_multiplier)
