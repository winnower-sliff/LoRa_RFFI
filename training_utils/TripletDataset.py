import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


# 自定义Dataset类，用于三元组生成
# 更新后的 TripletDataset 类
class TripletDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.dev_range = np.unique(labels)

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
            torch.tensor(anchor, dtype=torch.float32),
            torch.tensor(positive, dtype=torch.float32),
            torch.tensor(negative, dtype=torch.float32),
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
