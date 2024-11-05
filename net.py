import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


# 自定义Dataset类，用于三元组生成
# 更新后的 TripletDataset 类
class TripletDataset(Dataset):
    def __init__(self, data, labels, dev_range):
        self.data = data
        self.labels = labels
        self.dev_range = dev_range

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        anchor = self.data[idx]
        anchor_label = self.labels[idx]

        # 选择与 anchor 相同标签的 positive 样本
        positive_idx = np.random.choice(np.where(self.labels == anchor_label)[0])
        while positive_idx == idx:  # 确保 positive 与 anchor 不同
            positive_idx = np.random.choice(np.where(self.labels == anchor_label)[0])
        positive = self.data[positive_idx]

        # 选择与 anchor 不同标签的 negative 样本
        negative_label = np.random.choice(
            [label for label in self.dev_range if label != anchor_label]
        )
        negative_idx = np.random.choice(np.where(self.labels == negative_label)[0])
        negative = self.data[negative_idx]

        # 返回 anchor, positive, negative 样本
        return (
            torch.tensor(anchor, dtype=torch.float32).unsqueeze(0),
            torch.tensor(positive, dtype=torch.float32).unsqueeze(0),
            torch.tensor(negative, dtype=torch.float32).unsqueeze(0),
        )


# 三元组损失函数
# 更新后的三元组损失函数
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.triplet_loss = nn.TripletMarginLoss(margin=margin, p=2)

    def forward(self, anchor, positive, negative):
        return self.triplet_loss(anchor, positive, negative)


# 定义残差块
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, first_layer=False):
        super(ResBlock, self).__init__()
        self.first_layer = first_layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=1)
        self.relu = nn.ReLU(inplace=True)
        if self.first_layer:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        if self.first_layer:
            residual = self.shortcut(x)
        out += residual
        out = self.relu(out)
        return out


# 特征提取模型
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3)
        self.layer1 = ResBlock(32, 32)
        self.layer2 = ResBlock(32, 32)
        self.layer3 = ResBlock(32, 64, first_layer=True)
        self.layer4 = ResBlock(64, 64)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64, 512)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = F.normalize(self.fc(x), p=2, dim=1)  # L2 正则化
        return x


# TripletNet类，用于创建三元组网络
class TripletNet(nn.Module):
    def __init__(self, margin=0.1):
        super(TripletNet, self).__init__()
        self.margin = margin
        self.embedding_net = FeatureExtractor()

    def forward(self, anchor, positive, negative):
        embedded_anchor = self.embedding_net(anchor)
        embedded_positive = self.embedding_net(positive)
        embedded_negative = self.embedding_net(negative)
        return embedded_anchor, embedded_positive, embedded_negative

    def triplet_loss(self, anchor, positive, negative):
        return TripletLoss.apply(anchor, positive, negative, self.margin)
