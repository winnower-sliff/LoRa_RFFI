# 剪枝后的模型
import torch
import torch.nn as nn
import torch.nn.functional as F
from net.net_DRSN import Shrinkage


class PrunableBasicBlock(nn.Module):
    expansion = 1  # 扩张比例

    def __init__(self, in_channels, out_channels, stride=1, reduction_ratio=1.0):
        super().__init__()

        # 应用缩减比例
        actual_out_channels = int(out_channels * reduction_ratio)

        self.shrinkage = Shrinkage(
            actual_out_channels, gap_size=(1, 1)
        )  # 收缩层，用于减少输出维度或进行其他形式的特征压缩
        # 残差函数部分
        self.residual_function = nn.Sequential(
            nn.Conv2d(
                in_channels,
                actual_out_channels,
                kernel_size=(3, 3),  # 卷积核大小
                stride=stride,  # 步长
                padding=(1, 1),  # 填充
                bias=False,  # 是否使用偏置项
            ),
            nn.BatchNorm2d(actual_out_channels),  # 批量归一化层
            nn.ReLU(inplace=True),  # 激活函数，原地操作节省内存
            nn.Conv2d(
                actual_out_channels,
                actual_out_channels * self.expansion,  # 输出通道数乘以扩张比例
                kernel_size=(3, 3),
                padding=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(actual_out_channels * self.expansion),
            self.shrinkage,  # 应用收缩层
        )
        # 捷径（shortcut）部分，用于直接连接输入和输出，以保持信息的流通
        self.shortcut = nn.Sequential()

        # 如果捷径的输出维度与残差函数的输出维度不一致
        # 使用1x1卷积来匹配维度
        if stride != 1 or in_channels != self.expansion * actual_out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    actual_out_channels *
                    self.expansion,  # 输入通道数转为输出通道数乘以扩张比例
                    kernel_size=1,  # 使用1x1卷积核
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(actual_out_channels * self.expansion),  # 批量归一化层
            )

    def forward(self, x):
        # 前向传播，将输入x通过残差函数和捷径后相加，再通过ReLU激活函数
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class PrunedRSNet(nn.Module):

    def __init__(self, r, block, num_block, in_channels):
        """剪枝后的RSNet模型
        Args:
            r: 剪枝比例列表，每个元素对应一层的保留比例
            in_channels: 输入通道数
        """
        super().__init__()

        self.reduction_ratios = r

        # 初始卷积层
        base_channels = 32
        self.out_channels = int(base_channels * r[0])
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

        # 创建各层，传入缩减比例
        self.conv2_x = self._make_layer(block, 32, num_block[0], stride=1, reduction_ratio=r[1])
        self.conv3_x = self._make_layer(block, 32, num_block[1], stride=2, reduction_ratio=r[2])
        self.conv4_x = self._make_layer(block, 64, num_block[2], stride=2, reduction_ratio=r[3])
        self.conv5_x = self._make_layer(block, 64, num_block[3], stride=2, reduction_ratio=r[4])
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        # 动态计算全连接层输入维度
        # 由于剪枝后最后一层的输出通道数会变化，我们需要动态计算
        self.fc_input_dim = int(64 * block.expansion * r[4])
        self.fc = nn.Linear(self.fc_input_dim, 512)

    def _make_layer(self, block, out_channels, num_blocks, stride, reduction_ratio=1.0):
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
            layers.append(block(self.in_channels, out_channels, stride, reduction_ratio))
            self.in_channels = int(
                out_channels * block.expansion * reduction_ratio
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

        # 确保全连接层输入维度匹配
        if output.shape[1] != self.fc_input_dim:
            print(f"全连接层输入维度匹配")
            # 如果维度不匹配，进行截断或填充
            if output.shape[1] > self.fc_input_dim:
                output = output[:, :self.fc_input_dim]
            else:
                # 填充到所需维度
                padding = torch.zeros(output.shape[0], self.fc_input_dim - output.shape[1],
                                      device=output.device)
                output = torch.cat([output, padding], dim=1)

        output = F.normalize(self.fc(output), p=2, dim=1)  # L2 正则化

        return output


def pruned_drsnet18(r, in_channels):
    """return a RsNet 18 object"""
    return PrunedRSNet(r, PrunableBasicBlock, [2, 2, 2, 2], in_channels)


def pruned_drsnet34(r, in_channels):
    """return a RsNet 34 object"""
    return PrunedRSNet(r, PrunableBasicBlock, [3, 4, 6, 3], in_channels)
