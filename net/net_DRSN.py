import torch
import torch.nn as nn
import torch.nn.functional as F


class Shrinkage(nn.Module):
    def __init__(self, channel, gap_size):
        super(Shrinkage, self).__init__()
        # 初始化自适应平均池化层，用于将输入的特征图尺寸调整为gap_size
        self.gap = nn.AdaptiveAvgPool2d(gap_size)
        # 初始化一个全连接层序列，包括线性变换、批量归一化、ReLU激活函数、另一个线性变换和Sigmoid激活函数
        self.fc = nn.Sequential(
            nn.Linear(channel, channel),  # 线性变换，输入和输出通道数相同
            nn.BatchNorm1d(channel),  # 批量归一化
            nn.ReLU(inplace=True),  # ReLU激活函数，原地修改数据
            nn.Linear(channel, channel),  # 另一个线性变换，输入和输出通道数相同
            nn.Sigmoid(),  # Sigmoid激活函数，将输出值压缩到0到1之间
        )

    def forward(self, x):
        x_raw = x
        x_abs = torch.abs(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        average = x  # 保留平均值
        x = self.fc(x)
        x = torch.mul(average, x)  # 将x与平均值相乘
        x = x.unsqueeze(2).unsqueeze(2)
        sub = x_abs - x
        n_sub = torch.max(sub, torch.zeros_like(sub))
        x = torch.mul(torch.sign(x_raw), n_sub)
        return x


class BasicBlock(nn.Module):
    expansion = 1  # 扩张比例

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.shrinkage = Shrinkage(
            out_channels, gap_size=(1, 1)
        )  # 收缩层，用于减少输出维度或进行其他形式的特征压缩
        # 残差函数部分
        self.residual_function = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(3, 3),  # 卷积核大小
                stride=stride,  # 步长
                padding=(1, 1),  # 填充
                bias=False,  # 是否使用偏置项
            ),
            nn.BatchNorm2d(out_channels),  # 批量归一化层
            nn.ReLU(inplace=True),  # 激活函数，原地操作节省内存
            nn.Conv2d(
                out_channels,
                out_channels * self.expansion,  # 输出通道数乘以扩张比例
                kernel_size=(3, 3),
                padding=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(out_channels * self.expansion),
            self.shrinkage,  # 应用收缩层
        )
        # 捷径（shortcut）部分，用于直接连接输入和输出，以保持信息的流通
        self.shortcut = nn.Sequential()

        # 如果捷径的输出维度与残差函数的输出维度不一致
        # 使用1x1卷积来匹配维度
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels *
                    self.expansion,  # 输入通道数转为输出通道数乘以扩张比例
                    kernel_size=1,  # 使用1x1卷积核
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * self.expansion),  # 批量归一化层
            )

    def forward(self, x):
        # 前向传播，将输入x通过残差函数和捷径后相加，再通过ReLU激活函数
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class RSNet(nn.Module):

    def __init__(self, block, num_block, in_channels):
        super().__init__()

        # 初始卷积层
        self.out_channels = 32
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                self.out_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True),
        )
        self.in_channels = self.out_channels

        # we use a different inputsize than the original paper
        # so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 32, num_block[0], stride=1)
        self.conv3_x = self._make_layer(block, 32, num_block[1], stride=2)
        self.conv4_x = self._make_layer(block, 64, num_block[2], stride=2)
        self.conv5_x = self._make_layer(block, 64, num_block[3], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64 * block.expansion, 512)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """创建rsnet层（这里的'层'并不等同于神经网络层，例如卷积层），一个层可能包含多个残差收缩块

        参数:
            block: 块类型，基本块或瓶颈块
            out_channels: 该层的输出深度通道数
            num_blocks: 每层的块数量
            stride: 该层第一个块的步长

        返回:
            返回一个rsnet层
        """

        # 每层有num_block个块，第一个块的步长可能为1或2，其他块的步长始终为1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = (
                out_channels * block.expansion
            )  # 更新输入通道数为输出通道数乘以块的扩展系数

        return nn.Sequential(*layers)  # 返回一个按顺序排列的层

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = self.flatten(output)
        output = F.normalize(self.fc(output), p=2, dim=1)  # L2 正则化

        return output


def drsnet18(in_channels):
    """return a RsNet 18 object"""
    return RSNet(BasicBlock, [2, 2, 2, 2], in_channels)


def drsnet34(in_channels):
    """return a RsNet 34 object"""
    return RSNet(BasicBlock, [3, 4, 6, 3], in_channels)
