from random import randint, random
from matplotlib import pyplot as plt
from numpy.random import standard_normal, uniform
import h5py
import numpy as np


def _convert_to_complex(data):
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
    data_complex = data[:, : round(num_col / 2)] + 1j * data[:, round(num_col / 2) :]
    return data_complex


dataset_name = "data"
labelset_name = "label"

file_path = "./4307data/data_2025.2.26.h5"

dev_range, pkt_range = np.arange(0, 6, dtype=int), np.arange(000, 200, dtype=int)
calibration = False


with h5py.File(file_path, "r") as f:
    # 加载标签并转换为从0开始的索引
    label = f[labelset_name][:]
    label = label.astype(int).T - 1

    # 计算标签范围和每个设备的数据包数量
    label_start, label_end = int(label[0]) + 1, int(label[-1]) + 1
    num_dev = label_end - label_start + 1
    num_pkt = len(label)
    num_pkt_per_dev = int(num_pkt / num_dev)

    # 打印数据集信息
    print(
        f"Dataset information: Dev {label_start} to Dev {label_end}, {num_pkt_per_dev} packets per device."
    )

    # 根据设备和数据包范围构建样本索引列表
    sample_index_list = []
    for dev_idx in dev_range:
        # 注意：这里假设pkt_range可以是一个列表或一个slice对象
        sample_index_dev = np.where(label == dev_idx)[0][pkt_range].tolist()
        sample_index_list.extend(sample_index_dev)
    # print(label)

    # 加载数据并根据需要应用校准
    data = f[dataset_name][sample_index_list]

    data = _convert_to_complex(data)

    # 根据索引列表选择相应的标签
    label = label[sample_index_list]

    # 计算频偏
    F_dream = 433.125 * 10**6

    F_Rx_wo_err = F_dream
    F_Rx_err = randint(0, int(F_Rx_wo_err * 0.01))

    fs = 1000 * 10**3
    N = 8192  # 信号长度
    # true_freq_offset = 123

    for i in range(data.shape[0]):
        signal = data[i]
        # 计算FFT
        fft_result = np.fft.fft(signal)
        frequencies = np.fft.fftfreq(N, 1 / fs)

        # 计算频谱质心
        magnitude = np.abs(fft_result)
        f_centroid = np.sum(frequencies * magnitude) / np.sum(magnitude)

        # print(f"真实频偏: {true_freq_offset/1e3:.2f} kHz")
        print(f"频谱质心估计的中心频率: {f_centroid/1e3:.2f} kHz\tDev: {label[i]}")

        # 绘制频谱
        plt.title(f"Frequency Offset: {f_centroid/1e3:.2f} kHz + Rx_Error kHz")
        plt.plot(frequencies, 20 * np.log10(np.abs(fft_result)))
        # plt.axvline(true_freq_offset, color="r", linestyle="--", label="True CFO")
        plt.axvline(f_centroid, color="g", linestyle="--", label="Estimated CFO")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude (dB)")
        plt.legend()
        plt.show()
