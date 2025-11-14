# net/net_mobilenet.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class InvertedResidual(nn.Module):
    """倒残差块 - MobileNet V2 核心组件"""
    def __init__(self, inp, oup, stride, expansion_factor):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        hidden_dim = inp * expansion_factor
        self.use_residual = self.stride == 1 and inp == oup

        layers = []
        # 如果扩展因子大于1，先进行通道扩展
        if expansion_factor != 1:
            layers.append(nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))

        # 深度卷积
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # 逐点卷积（线性）
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)



class MobileNet(nn.Module):
    """MobileNet 特征提取器"""
    def __init__(self, in_channels, width_multiplier=1.0):
        super(MobileNet, self).__init__()
        self.width_multiplier = width_multiplier
        
        def conv_bn(inp, oup, stride):
            '''标准卷积块：卷积 + 批归一化 + ReLU'''
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True)
            )
        
        def conv_dw(inp, oup, stride):
            '''深度可分离卷积块：深度卷积 + 逐点卷积'''
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU6(inplace=True),
                
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True),
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

        # 全局平均池化层：将特征图压缩为1x1
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # 展平层：将多维特征转换为一维向量
        self.flatten = nn.Flatten()
        # 全连接层：将特征映射到512维特征空间
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
