# TripletNet类，用于创建三元组网络
import numpy as np
import torch
import torch.nn as nn

from core.config import Config
from net.net_DRSN import drsnet18
from net.net_original import FeatureExtractor
from net.net_prune import PrunedRSNet
from training_utils.TripletDataset import TripletLoss


class TripletNet(nn.Module):
    def __init__(self, net_type, in_channels, margin=0.1, config: Config = None):
        super(TripletNet, self).__init__()
        self.margin = margin
        if net_type == 0:
            self.embedding_net = FeatureExtractor(in_channels=in_channels)
        elif net_type == 1:
            self.embedding_net = drsnet18(in_channels=in_channels)
        elif net_type == 2:
            # 加载剪枝率
            r = np.loadtxt(config.custom_pruning_file, delimiter=",")
            r = [1 - x for x in r]
            self.embedding_net = PrunedRSNet(r, in_channels=in_channels)

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
