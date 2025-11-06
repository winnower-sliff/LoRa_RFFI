# 剪枝后的模型
import torch
from torch import nn
import torch.nn.functional as F


class Shrinkage(nn.Module):
    def __init__(self, channel, gap_size):
        super(Shrinkage, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(gap_size)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel),
            nn.BatchNorm1d(channel),
            nn.ReLU(inplace=True),
            nn.Linear(channel, channel),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x_raw = x
        x_abs = torch.abs(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        average = x
        x = self.fc(x)
        x = torch.mul(average, x)
        x = x.unsqueeze(2).unsqueeze(2)
        sub = x_abs - x
        n_sub = torch.max(sub, torch.zeros_like(sub))
        x = torch.mul(torch.sign(x_raw), n_sub)
        return x


class PrunableBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, reduction_ratio=1.0):
        super().__init__()

        # 应用缩减比例
        actual_out_channels = int(out_channels * reduction_ratio)

        self.shrinkage = Shrinkage(actual_out_channels, gap_size=(1, 1))

        # 残差函数部分
        self.residual_function = nn.Sequential(
            nn.Conv2d(
                in_channels,
                actual_out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(actual_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                actual_out_channels,
                actual_out_channels * self.expansion,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(actual_out_channels * self.expansion),
            self.shrinkage,
        )

        # 捷径部分
        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != actual_out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    actual_out_channels * self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(actual_out_channels * self.expansion),
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class PrunedRSNet(nn.Module):
    def __init__(self, r, in_channels=1, num_classes=10):
        """剪枝后的RSNet模型

        Args:
            r: 剪枝比例列表，每个元素对应一层的保留比例
            num_classes: 分类类别数
            in_channels: 输入通道数
        """
        super().__init__()

        self.reduction_ratios = r

        # 初始卷积层
        base_channels = 32
        self.conv1_out_channels = int(base_channels * r[0])
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, self.conv1_out_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.conv1_out_channels),
            nn.ReLU(inplace=True),
        )

        self.in_channels = self.conv1_out_channels

        # 创建各层，传入缩减比例
        self.conv2_x = self._make_layer(PrunableBasicBlock, 32, 2, stride=1, reduction_ratio=r[1])
        self.conv3_x = self._make_layer(PrunableBasicBlock, 32, 2, stride=2, reduction_ratio=r[2])
        self.conv4_x = self._make_layer(PrunableBasicBlock, 64, 2, stride=2, reduction_ratio=r[3])
        self.conv5_x = self._make_layer(PrunableBasicBlock, 64, 2, stride=2, reduction_ratio=r[4])

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        # 动态计算全连接层输入维度
        # 由于剪枝后最后一层的输出通道数会变化，我们需要动态计算
        self.fc_input_dim = int(64 * PrunableBasicBlock.expansion * r[4])
        self.fc = nn.Linear(self.fc_input_dim, 512)

        # 分类头
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
            nn.Softmax(dim=1)
        )

    def _make_layer(self, block, out_channels, num_blocks, stride, reduction_ratio=1.0):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride, reduction_ratio))
            self.in_channels = int(out_channels * block.expansion * reduction_ratio)

        return nn.Sequential(*layers)

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
            # 如果维度不匹配，进行截断或填充
            if output.shape[1] > self.fc_input_dim:
                output = output[:, :self.fc_input_dim]
            else:
                # 填充到所需维度
                padding = torch.zeros(output.shape[0], self.fc_input_dim - output.shape[1],
                                      device=output.device)
                output = torch.cat([output, padding], dim=1)

        output = F.normalize(self.fc(output), p=2, dim=1)
        output = self.classifier(output)
        return output