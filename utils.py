# 模型加载函数
import numpy as np
import torch
from LoadDataset import LoadDataset
from net.TripletNet import TripletNet
from signal_trans import TimeFrequencyTransformer


def load_model(file_path, net_type, generate_type, weights_only=True):
    """从指定路径加载模型"""
    model = TripletNet(net_type=net_type, in_channels=1 if generate_type == 0 else 2)
    model.load_state_dict(torch.load(file_path, weights_only=weights_only))
    model.eval()
    # print(f"Model loaded from {file_path}")
    return model


# 数据预处理函数
def generate_spectrogram(data, generate_type, wst_j=6, wst_q=6):
    """
    数据预处理方式选择

    - PROPRECESS_TYPE : 0 信道独立的STFT
    - PROPRECESS_TYPE : 1 小波散射变换
    """
    # TF_Transformer = TimeFrequencyTransformer()
    if generate_type == 0:
        data = TimeFrequencyTransformer.generate_stft_channel(data)
    if generate_type == 1:
        data = TimeFrequencyTransformer.generate_WST(data, J=wst_j, Q=wst_q)
    return data


def load_data(file_path, dev_range, pkt_range) -> tuple[np.ndarray, np.ndarray]:
    """根据参数加载指定数据"""
    # LoadDatasetObj = LoadDataset()
    data, label = LoadDataset.load_iq_samples(file_path, dev_range, pkt_range)

    return data, label

# 数据预处理
def load_generate(file_path, dev_range, pkt_range, generate_type):
    """加载 & 预处理"""
    data, label = load_data(file_path, dev_range, pkt_range)
    data = generate_spectrogram(data, generate_type)

    # 数据三元组化并转换为张量类型
    data = [torch.tensor(data).float() ]

    return label, data

# 数据预处理
def load_generate_triplet(file_path, dev_range, pkt_range, generate_type):
    """加载 & 预处理 & data三元组化"""
    data, label = load_data(file_path, dev_range, pkt_range)
    data = generate_spectrogram(data, generate_type)

    # 数据三元组化并转换为张量类型
    triplet_data = [data, data, data]
    triplet_data = [torch.tensor(x).float() for x in triplet_data]

    return label, triplet_data

