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
PRUNED_DATA_DIR = "./experiments/pruning_results/"  # 剪枝相关文件存放
CUSTOM_PRUNING_FILE = os.path.join(PRUNED_DATA_DIR, "1-pr.csv")
# PCA相关的配置
PCA_DATA_DIR = "./experiments/pca_results/"
PCA_FILE_INPUT = os.path.join(PCA_DATA_DIR, "teacher_feats.npz")
PCA_FILE_OUTPUT = os.path.join(PCA_DATA_DIR, "pca_16.npz")

PCA_DIM_TRAIN = 8  # 训练时PCA的维度
PCA_DIM_TEST = 8    # 测试时PCA的维度

if not os.path.exists(PRUNED_DATA_DIR):
    os.makedirs(PRUNED_DATA_DIR)
if not os.path.exists(PCA_DATA_DIR):
    os.makedirs(PCA_DATA_DIR)


# 定义运行模式的枚举
class Mode:
    TRAIN = "train"
    CLASSIFICATION = "classification"
    ROGUE_DEVICE_DETECTION = "rogue_device_detection"
    PRUNE = "prune"
    DISTILLATION = "distillation"
    TEST = "test"


# 定义网络类型枚举
class NetworkType(Enum):
    RESNET = "ResNet"      # 残差网络
    DRSN = "Drsn"       # 深度残差网路
    MobileNetV1 = "MobileNetV1"  # MobileNetV1网络
    MobileNetV2 = "MobileNetV2"  # MobileNetV2网络
    LightNetV1 = "LightNetV1"     # MobileNetV1改进网络
    LightNetV2 = "LightNetV2"     # LightNetV1改进网络


# 定义预处理类型枚举
class PreprocessType(Enum):
    IQ = ("IQ", 2)  # IQ数据直接使用，2个通道（I和Q）
    STFT = ("STFT", 1)  # 短时傅里叶变换，1个通道（幅度）
    WST = ("WST", 2)  # 小波散射变换，2个通道（实部和虚部）

    def __init__(self, name, in_channels):
        self._value_ = name
        self.in_channels = in_channels



# 配置类，用于存储全局配置参数
class Config:
    def __init__(self, mode):
        # 设置模式
        self.mode = mode
        # 设置网络类型
        self.NET_TYPE = NetworkType.LightNetV1
        # 教师网络类型, 学生网络类型
        self.TEACHER_NET_TYPE = NetworkType.RESNET
        self.STUDENT_NET_TYPE = NetworkType.LightNetV1
        # 数据预处理类型
        self.PROPRECESS_TYPE = PreprocessType.STFT
        # 0 for all, 1 for only prune, 2 for only test
        self.PRUNE_MODE = 0
        # 0 for all, 1 for only distillate, 2 for only Fine-tuning, 3 for only test, 4 for only rogue
        self.DISTILLATE_MODE = 4
        # 蒸馏训练是否使用PCA
        self.IS_PCA_TRAIN = True
        # 测试时是否使用PCA
        self.IS_PCA_TEST = True
        # 我们需要一个新的训练文件吗？
        self.new_file_flag = 1

        self.TEST_LIST = [1, 5, 10, 20, 50, 100, 150, 200, 250, 300]
        self.WST_J = 6
        self.WST_Q = 6


        # 后续设置

        if self.PROPRECESS_TYPE == PreprocessType.STFT:
            self.PPS_FOR = "stft"
            self.MODEL_DIR= f"./model/{self.PPS_FOR}/{self.NET_TYPE.value}/"
            self.TEACHER_MODEL_DIR = f"./model/{self.PPS_FOR}/{self.TEACHER_NET_TYPE.value}/"
            self.STUDENT_MODEL_DIR = f"./model/{self.PPS_FOR}/{self.STUDENT_NET_TYPE.value}/"
            self.filename_train_prepared_data = f"train_data_{self.PPS_FOR}.h5"
        else:
            self.PPS_FOR = "wst"
            self.ORIGIN_MODEL_DIR_PATH = (  # 更正变量名，使其更加明确
                f"./model/{self.PPS_FOR}_j{self.WST_J}q{self.WST_Q}/{self.NET_TYPE.value}/"  # 添加.value访问枚举值
            )
            self.filename_train_prepared_data = (
                f"train_data_{self.PPS_FOR}_j{self.WST_J}q{self.WST_Q}.h5"
            )

        # 创建模型目录
        if not os.path.exists(self.MODEL_DIR):
            os.makedirs(self.MODEL_DIR)
        if not os.path.exists(self.TEACHER_MODEL_DIR):
            os.makedirs(self.TEACHER_MODEL_DIR)
        if not os.path.exists(self.STUDENT_MODEL_DIR):
            os.makedirs(self.STUDENT_MODEL_DIR)


def set_seed(seed=42):
    """设置随机种子以确保实验可重现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False