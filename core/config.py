"""配置管理模块"""
import os
import random

import numpy as np
import torch
from enum import Enum


# 设备配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 剪枝相关的配置
H_VAL = 10  # 剪枝的粒度, 即决定剪枝的激进程度
PRUNED_OUTPUT_DIR = "./pruning_results/"  # 剪枝相关文件存放
CUSTOM_PRUNING_FILE = os.path.join(PRUNED_OUTPUT_DIR, "1-pr.csv")


# 定义运行模式的枚举
class Mode:
    TRAIN = "train"
    CLASSIFICATION = "classification"
    ROGUE_DEVICE_DETECTION = "rogue_device_detection"
    PRUNE = "prune"
    DISTILLATION = "distillation"


# 定义网络类型枚举
class NetworkType(Enum):
    RESNET = "ResNet"      # 残差网络
    DRSN = "Drsn"       # 深度残差网路
    MobileNetV1 = "MobileNetV1"  # MobileNetV1网络
    MobileNetV2 = "MobileNetV2"  # MobileNetV1网络


# 定义预处理类型枚举
class PreprocessType(Enum):
    STFT = 0          # 短时傅里叶变换
    WST = 1           # 小波散射变换



# 配置类，用于存储全局配置参数
class Config:
    def __init__(self, mode):
        # 设置模式
        self.mode = mode
        # 设置网络类型
        self.NET_TYPE = NetworkType.RESNET.value
        self.TEACHER_NET_TYPE = NetworkType.RESNET.value
        self.STUDENT_NET_TYPE = NetworkType.MobileNetV2.value
        # 0 for stft, 1 for wst
        self.PROPRECESS_TYPE = PreprocessType.STFT.value
        # 0 for all, 1 for only prune, 2 for only test
        self.prune_mode = 0
        # 0 for all, 1 for only distillate, 2 for only test
        self.distillate_mode = 0
        # 我们需要一个新的训练文件吗？
        self.new_file_flag = 1

        self.TEST_LIST = [1, 5, 10, 20, 50, 100, 150, 200, 250, 300]
        self.WST_J = 6
        self.WST_Q = 6


        # 后续设置

        if self.PROPRECESS_TYPE == PreprocessType.STFT.value:
            self.PPS_FOR = "stft"
            self.MODEL_DIR= f"./model/{self.PPS_FOR}/{self.NET_TYPE}/"
            self.TEACHER_MODEL_DIR = f"./model/{self.PPS_FOR}/{self.TEACHER_NET_TYPE}/"
            self.STUDENT_MODEL_DIR = f"./model/{self.PPS_FOR}/{self.STUDENT_NET_TYPE}/"
            self.filename_train_prepared_data = f"train_data_{self.PPS_FOR}.h5"
        else:
            self.PPS_FOR = "wst"
            self.ORIGIN_MODEL_DIR_PATH = (
                f"./model/{self.PPS_FOR}_j{self.WST_J}q{self.WST_Q}/{self.NET_TYPE}/"
            )
            self.filename_train_prepared_data = (
                f"train_data_{self.PPS_FOR}_j{self.WST_J}q{self.WST_Q}.h5"
            )

        if not os.path.exists(self.MODEL_DIR):
            os.makedirs(self.MODEL_DIR)


def set_seed(seed=42):
    """设置随机种子以确保实验可重现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
