"""配置管理模块"""
import os
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
    PRUNING = "pruning"


# 定义网络类型枚举
class NetworkType(Enum):
    ORIGINAL = 0      # 原始网络
    WAVELET = 1       # 小波变换网络
    PRUNE = 2         # 剪枝模型


# 定义预处理类型枚举
class PreprocessType(Enum):
    STFT = 0          # 短时傅里叶变换
    WST = 1           # 小波散射变换


# 定义剪枝类型枚举
class PruneType(str, Enum):
    l2 = "l2"
    fpgm = "fpgm"


# 配置类，用于存储全局配置参数
class Config:
    def __init__(self):
        # 不要 net_0 & pps_1
        # 0 for origin, 1 for drsn, 2 for prune
        self.NET_TYPE = NetworkType.PRUNE.value
        # 0 for stft, 1 for wst
        self.PROPRECESS_TYPE = PreprocessType.STFT.value
        # 我们需要一个新的训练文件吗？
        self.new_file_flag = 1

        self.TEST_LIST = [1, 5, 10, 20, 50, 100, 150, 200, 250, 300]
        self.WST_J = 6
        self.WST_Q = 6


        # 后续设置
        if self.NET_TYPE == NetworkType.ORIGINAL.value:
            self.NET_NAME = "origin"
        elif self.NET_TYPE == NetworkType.WAVELET.value:
            self.NET_NAME = "drsn"
        elif self.NET_TYPE == NetworkType.PRUNE.value:
            self.NET_NAME = "prune"

        if self.PROPRECESS_TYPE == PreprocessType.STFT.value:
            self.PPS_FOR = "stft"
            self.MODEL_DIR_PATH = f"./model/{self.PPS_FOR}/{self.NET_NAME}/"
            self.filename_train_prepared_data = f"train_data_{self.PPS_FOR}.h5"
        else:
            self.PPS_FOR = "wst"
            self.MODEL_DIR_PATH = (
                f"./model/{self.PPS_FOR}_j{self.WST_J}q{self.WST_Q}/{self.NET_NAME}/"
            )
            self.filename_train_prepared_data = (
                f"train_data_{self.PPS_FOR}_j{self.WST_J}q{self.WST_Q}.h5"
            )

        if not os.path.exists(self.MODEL_DIR_PATH):
            os.makedirs(self.MODEL_DIR_PATH)
