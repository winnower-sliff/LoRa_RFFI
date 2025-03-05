from math import ceil, floor
from matplotlib import pyplot as plt
import numpy as np
from numpy import sum, sqrt
from numpy.random import standard_normal, uniform

import scipy.signal as signal
import h5py
from kymatio import Scattering1D
from tqdm import tqdm


# Dataset loader class for IQ samples
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

    def load_iq_samples(self, file_path, dev_range, pkt_range, calibration=False):
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
            data = f[self.dataset_name][sample_index_list]
            if calibration:
                data = self._convert_to_complex_with_calibration(data)
            else:
                data = self._convert_to_complex(data)

            # 根据索引列表选择相应的标签
            label = label[sample_index_list]

        return data, label


# Additive White Gaussian Noise (AWGN) Function
def awgn(data, snr_range):
    """
    向输入数据中的每个数据包添加加性高斯白噪声（AWGN）。

    参数:
    data (numpy.ndarray): 二维数组，其中每行代表一个数据包（复数信号）。
    snr_range (tuple or list): 包含SNR范围（以dB为单位）的元组或列表。

    返回:
    numpy.ndarray: 包含添加了噪声的数据包的新数据数组。
    """
    pkt_num = data.shape[0]
    SNRdB = uniform(snr_range[0], snr_range[-1], pkt_num)
    for pktIdx in range(pkt_num):
        s = data[pktIdx]
        SNR_linear = 10 ** (SNRdB[pktIdx] / 10)
        P = sum(abs(s) ** 2) / len(s)
        N0 = P / SNR_linear
        n = sqrt(N0 / 2) * (standard_normal(len(s)) + 1j * standard_normal(len(s)))
        data[pktIdx] = s + n

    return data


class TimeFrequencyTransformer:
    """将时域信号转化为时频图，包括通道独立频谱图和小波散射频谱图的生成"""

    def generate_stft_channel(self, data, win_len=256, overlap=128):
        """
        将一批IQ信号样本转换为通道独立频谱图。

        参数:
        data (ndarray): 输入的IQ样本数据，每一行代表一个IQ样本。
        win_len (int, 可选): 窗口长度，默认为256。
        overlap (int, 可选): 窗口重叠长度，默认为128。

        返回:
        ndarray: 转换后的通道独立频谱图，形状为 (样本数量, 频谱图行数, 频谱图列数, 1)。
        """

        def _normalization(data):
            """对输入信号进行归一化处理"""
            s_norm = np.zeros(data.shape, dtype=complex)
            for i in range(data.shape[0]):
                sig_amplitude = np.abs(data[i])
                rms = np.sqrt(np.mean(sig_amplitude**2))
                s_norm[i] = data[i] / rms
            return s_norm

        def _spec_crop(x):
            """裁剪生成的通道独立频谱图"""
            num_row = x.shape[0]
            x_cropped = x[round(num_row * 0.3) : round(num_row * 0.7)]
            return x_cropped

        def _gen_single_channel_ind_spectrogram(sig, win_len=256, overlap=128):
            """根据设定的窗口长度和重叠长度，将IQ样本转换为单个通道独立频谱图"""
            # Short-time Fourier transform (STFT).
            f, t, spec = signal.stft(
                sig,
                window="boxcar",
                nperseg=win_len,
                noverlap=overlap,
                nfft=win_len,
                return_onesided=False,
                padded=False,
                boundary=None,
            )

            # FFT shift to adjust the central frequency.
            spec = np.fft.fftshift(spec, axes=0)

            # Generate channel independent spectrogram.
            chan_ind_spec = spec[:, 1:] / spec[:, :-1]

            # Take the logarithm of the magnitude.
            chan_ind_spec_amp = np.log10(np.abs(chan_ind_spec) ** 2)
            return chan_ind_spec_amp

        # Normalize the IQ samples.
        data = _normalization(data)

        # Calculate the size of channel independent spectrograms.
        num_sample = data.shape[0]
        num_row = int(256 * 0.4)
        num_column = int(np.floor((data.shape[1] - 256) / 128 + 1) - 1)
        data_channel_ind_spec = np.zeros([num_sample, 1, num_row, num_column])

        # Convert each packet (IQ samples) to a channel independent spectrogram.
        for i in tqdm(range(num_sample)):
            chan_ind_spec_amp = _gen_single_channel_ind_spectrogram(
                data[i], win_len, overlap
            )
            chan_ind_spec_amp = _spec_crop(chan_ind_spec_amp)
            data_channel_ind_spec[i, 0, :, :] = chan_ind_spec_amp

        return data_channel_ind_spec

    def generate_WST(self, data, J=6, Q=6):
        """
        将批量IQ样本转换为小波散射特征图。

        参数:
        data (ndarray): 输入的IQ样本数据，其中每一行代表一个IQ样本。
        J (int, 可选): 小波散射的尺度，默认为6。
        Q (int, 可选): 每倍频程的滤波器数量，默认为8。

        返回:
        ndarray: 一个三维数组，包含所有IQ样本对应的小波散射特征图。
                形状为 (样本数量, 时间点数, 特征数)。
        """
        num_sample, sample_length = data.shape

        # 初始化小波散射
        scattering = Scattering1D(J=J, Q=Q, shape=(sample_length,))

        meta = scattering.meta()
        order0 = np.where(meta["order"] == 0)
        order1 = np.where(meta["order"] == 1)[0]
        order2 = np.where(meta["order"] == 2)

        # 计算小波散射输出形状
        scatter_example = scattering(np.real(data[0]))  # 使用第一个样本计算输出
        num_row, num_column = scatter_example.shape  # 确定时间点数和特征数

        # 初始化存储矩阵
        s, e = floor(0.1 * num_row), floor(0.4 * num_row)
        data_wav_spec = np.zeros((num_sample, 2, len(order1), num_column - 1))

        for i in tqdm(range(num_sample)):
            sig = np.asarray(data[i])  # 当前样本信号

            # 分离复数信号的实部和虚部
            real_part = np.real(sig)
            imag_part = np.imag(sig)

            # 分别计算实部和虚部的小波散射
            scatter_real = scattering(real_part)
            scatter_imag = scattering(imag_part)

            # # 将实部和虚部结果组合（例如计算幅度）
            scatter_combined = np.stack((scatter_real, scatter_imag), axis=0)
            # scatter_combined = scatter_combined[:, s:e]
            # # Generate channel independent spectrogram.
            # scatter_combined = scatter_combined[:, :, 1:] / scatter_combined[:, :, :-1]
            # scatter_combined = np.log10(np.abs(scatter_combined) ** 2)

            """绘制结果，不要删除！！！"""
            # for i in range(2):
            #     plt.figure(figsize=(8, 8))
            #     plt.subplot(3, 1, 1)
            #     plt.plot(scatter_combined[i][order0][0])
            #     plt.title("Zeroth-order scattering")
            #     plt.subplot(3, 1, 2)
            #     plt.imshow(scatter_combined[i][order1], aspect="auto")
            #     plt.title("First-order scattering")
            #     plt.subplot(3, 1, 3)
            #     plt.imshow(scatter_combined[i][order2], aspect="auto")
            #     plt.title("Second-order scattering")
            #     plt.tight_layout()
            #     print(
            #         scatter_combined[i][order1].shape, scatter_combined[i][order2].shape
            #     )
            #     plt.show()

            scatter_combined = scatter_combined[:, order1]
            scatter_combined = scatter_combined[:, :, 1:] / scatter_combined[:, :, :-1]

            # for i in range(2):
            #     plt.figure()

            #     plt.imshow(scatter_combined[i], aspect="auto")
            #     plt.title("First-order scattering")

            #     plt.show()
            # 存储结果
            data_wav_spec[i] = scatter_combined
        return data_wav_spec
