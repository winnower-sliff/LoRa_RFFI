# TripletNet类，用于创建三元组网络
import numpy as np
import torch
import torch.nn as nn

from modes.net_MobileNet import mobilenet
from net.net_DRSN import drsnet18
from net.net_resnet import FeatureExtractor
from net.net_prune import pruned_drsnet18
from training_utils.TripletDataset import TripletLoss


class TripletNet(nn.Module):
    def __init__(self, net_type, in_channels,
                 custom_pruning_file=None,
                 use_pytorch_prune=True,   # 新增：是否使用PyTorch原生剪枝
                pruning_rates=None,      # 新增：PyTorch剪枝率
                margin=0.1
        ):
        super(TripletNet, self).__init__()
        self.margin = margin
        if net_type == 0:
            self.embedding_net = FeatureExtractor(in_channels=in_channels)
        elif net_type == 1:
            self.embedding_net = drsnet18(in_channels=in_channels)
        elif net_type == 2:
            if use_pytorch_prune:
                # PyTorch原生剪枝模式：先创建原始网络，稍后应用剪枝
                self.embedding_net = drsnet18(in_channels=in_channels)
                self.pruning_rates = pruning_rates if pruning_rates is not None else []
            else:
                # 加载剪枝率
                r = np.loadtxt(custom_pruning_file, delimiter=",")
                r = [1 - x for x in r]
                self.embedding_net = pruned_drsnet18(r, in_channels=in_channels)
        elif net_type == 3:  # 添加 MobileNet 支持
            self.embedding_net = mobilenet(in_channels=in_channels, width_multiplier=0.25)
        # 其他网络类型可以继续添加...

    def forward(self, anchor, positive, negative):
        embedded_anchor = self.embedding_net(anchor)
        embedded_positive = self.embedding_net(positive)
        embedded_negative = self.embedding_net(negative)
        return embedded_anchor, embedded_positive, embedded_negative

    def triplet_loss(self, anchor, positive, negative):
        loss_fn = TripletLoss(margin=self.margin)
        return loss_fn(anchor, positive, negative)

    def predict(self, anchor):
        with torch.no_grad():
            return self.embedding_net(anchor)
