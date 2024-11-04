import h5py
import numpy as np

import torch
from torch.utils.data import Dataset

from net import TripletNet


# 模型保存函数
def save_model(model, file_path="Extractor.pth"):
    """保存模型到指定路径"""
    torch.save(model.state_dict(), file_path)
    print(f"Model saved to {file_path}")


# 模型加载函数
def load_model(file_path="Extractor.pth"):
    """从指定路径加载模型"""
    model = TripletNet()
    model.load_state_dict(torch.load(file_path))
    model.eval()
    print(f"Model loaded from {file_path}")
    return model


# Dataset loader class for IQ samples
class LoadDataset:
    def __init__(self):
        self.dataset_name = "data"
        self.labelset_name = "label"

    def _convert_to_complex(self, data):
        """Convert the loaded data to complex IQ samples."""
        num_row = data.shape[0]
        num_col = data.shape[1]
        data_complex = np.zeros([num_row, round(num_col / 2)], dtype=complex)
        data_complex = (
            data[:, : round(num_col / 2)] + 1j * data[:, round(num_col / 2) :]
        )
        return data_complex

    def load_iq_samples(self, file_path, dev_range, pkt_range):
        """
        从数据集中加载IQ样本。  
        
        输入参数：  
            FILE_PATH 是数据集的路径。  
            DEV_RANGE 指定了要加载的设备范围。  
            PKT_RANGE 指定了要加载的数据包范围。  
        
        返回值：  
            DATA 是加载的复数IQ样本。  
            LABEL 是每个接收到的数据包的真实标签。
        """
        with h5py.File(file_path, "r") as f:
            label = f[self.labelset_name][:]
            label = label.astype(int).T - 1

            label_start, label_end = int(label[0]) + 1, int(label[-1]) + 1
            num_dev = label_end - label_start + 1
            num_pkt = len(label)
            num_pkt_per_dev = int(num_pkt / num_dev)

            print(
                f"Dataset information: Dev {label_start} to Dev {label_end}, {num_pkt_per_dev} packets per device."
            )

            sample_index_list = []
            for dev_idx in dev_range:
                sample_index_dev = np.where(label == dev_idx)[0][pkt_range].tolist()
                sample_index_list.extend(sample_index_dev)

            data = f[self.dataset_name][sample_index_list]
            data = self._convert_to_complex(data)
            label = label[sample_index_list]

        return data, label


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
        anchor, positive, negative = self._get_triplet()
        return (
            torch.tensor(anchor, dtype=torch.float32).unsqueeze(0),
            torch.tensor(positive, dtype=torch.float32).unsqueeze(0),
            torch.tensor(negative, dtype=torch.float32).unsqueeze(0),
        )

    def _get_triplet(self):
        """随机选择一个三元组 (anchor, positive, negative)"""
        anchor_label = np.random.choice(self.dev_range)
        positive_label = anchor_label
        negative_label = np.random.choice(
            [label for label in self.dev_range if label != anchor_label]
        )

        anchor = self._sample_by_label(anchor_label)
        positive = self._sample_by_label(positive_label)
        negative = self._sample_by_label(negative_label)

        return anchor, positive, negative

    def _sample_by_label(self, label):
        """从数据中根据标签选择样本"""
        indices = np.where(self.labels == label)[0]
        idx = np.random.choice(indices)
        return self.data[idx]
