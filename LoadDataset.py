# Dataset loader class for IQ samples
from random import uniform
import h5py
import numpy as np


class LoadDataset:
    def __init__(self):
        self.dataset_name = "data"
        self.labelset_name = "label"

    def _convert_to_complex(self, data):
        """
        将加载的数据转换为复数IQ样本，不应用任何校准。

        参数:
        data (numpy.ndarray): 输入的二维数组数据，其中I和Q样本交替排列。

        返回:
        numpy.ndarray: 转换后的复数IQ样本数组。
        """
        num_row = data.shape[0]
        num_col = data.shape[1]
        data_complex = np.zeros([num_row, round(num_col / 2)], dtype=complex)
        data_complex = (
            data[:, : round(num_col / 2)] + 1j * data[:, round(num_col / 2) :]
        )
        return data_complex

    def _convert_to_complex_with_calibration(self, data):
        """
        将加载的数据转换为复数IQ样本，并应用随机校准系数调整I和Q分量。

        参数:
        data (numpy.ndarray): 输入的二维数组数据，其中I和Q样本交替排列。

        返回:
        numpy.ndarray: 经过校准后转换的复数IQ样本数组。

        注意:
        校准系数包括随机选择的偏移量和在一定范围内均匀分布的乘法系数。
        """
        num_row = data.shape[0]
        num_col = data.shape[1]
        data_complex = np.zeros([num_row, round(num_col / 2)], dtype=complex)

        # 校准参数生成
        coeffs_a = np.arange(-0.05, 0.06, 0.01)
        coeffs_a_r_I = np.random.choice(coeffs_a, round(num_col / 2))
        coeffs_a_r_Q = np.random.choice(coeffs_a, round(num_col / 2))
        coeffs_m_I = uniform(0.95, 1.05, round(num_col / 2))
        coeffs_m_Q = uniform(0.95, 1.05, round(num_col / 2))

        # 实现校准
        tt1 = data[:, : round(num_col / 2)] * coeffs_m_I + coeffs_a_r_I
        tt2 = data[:, round(num_col / 2) :] * coeffs_m_Q + coeffs_a_r_Q
        data_complex = tt1 + 1j * tt2
        return data_complex

    def load_iq_samples(
        self,
        file_path: str,
        dev_range: list[int],
        pkt_range: list[int] = None,
        calibration=False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        从指定路径的数据集中加载IQ样本，并可选地应用校准。

        参数:
            file_path (str): 数据集的路径。
            dev_range (list of int): 要加载的设备ID范围。
            pkt_range (list of int or slice): 要加载的数据包索引范围。
            calibration (bool): 是否应用校准到IQ样本上，默认为False。

        返回:
            tuple: 包含两个元素，分别是
                - data (numpy.ndarray): 加载并可能经过校准的复数IQ样本。
                - label (numpy.ndarray): 每个接收到的数据包的真实标签（整数）。
        """
        with h5py.File(file_path, "r") as f:
            # 加载标签并转换为从0开始的索引
            label = f[self.labelset_name][:]
            label = label.astype(int).T - 1

            if pkt_range is not None:  # 如果指定了读取范围/数量
                # 根据设备和数据包范围构建样本索引列表
                sample_index_list = []
                for dev_idx in dev_range:
                    # 注意：这里假设pkt_range可以是一个列表或一个slice对象
                    sample_index_dev = np.where(label == dev_idx)[0][pkt_range].tolist()
                    sample_index_list.extend(sample_index_dev)
                # print(label)

                # 加载数据并根据需要应用校准
                data = f[self.dataset_name][sample_index_list]
                label = label[sample_index_list]
            else:  # 没有指定就全读
                data = f[self.dataset_name][:]

            # 是否应用校准到IQ样本上
            if calibration:
                data = self._convert_to_complex_with_calibration(data)
            else:
                data = self._convert_to_complex(data)

            # 打印数据集信息
            num_pkt = data.shape[0] // dev_range.shape[0]
            print(
                f"Dataset information: Devs: {{{str(dev_range)[1:-1]}}}, {num_pkt} packets per device."
            )
        return data, label
