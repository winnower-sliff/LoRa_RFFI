from math import ceil, floor
from matplotlib import pyplot as plt
import numpy as np
from numpy import sum, sqrt
from numpy.random import standard_normal, uniform

import scipy.signal as signal
import h5py
from kymatio import Scattering1D
from kymatio.scattering1d.filter_bank import scattering_filter_factory

from tqdm import tqdm
import pywt


def db1_filter(J, shape):
    wavelet = pywt.Wavelet("db1")
    filters = scattering_filter_factory(shape=shape, J=J, wavelet=wavelet)
    return filters


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

    def _convert_to_complex_with_calibration(self, data):
        """Convert the loaded data to complex IQ samples."""
        num_row = data.shape[0]
        num_col = data.shape[1]
        data_complex = np.zeros([num_row, round(num_col / 2)], dtype=complex)

        coeffs_a = np.arange(-0.05, 0.06, 0.01)
        coeffs_a_r_I = np.random.choice(coeffs_a, round(num_col / 2))
        coeffs_a_r_Q = np.random.choice(coeffs_a, round(num_col / 2))
        coeffs_m_I = uniform(0.95, 1.05, round(num_col / 2))
        coeffs_m_Q = uniform(0.95, 1.05, round(num_col / 2))

        tt1 = data[:, : round(num_col / 2)] * coeffs_m_I + coeffs_a_r_I
        tt2 = data[:, round(num_col / 2) :] * coeffs_m_Q + coeffs_a_r_Q
        data_complex = tt1 + 1j * tt2
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

    def load_iq_samples_with_calibration(self, file_path, dev_range, pkt_range):
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
            data = self._convert_to_complex_with_calibration(data)
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


class ChannelIndSpectrogram:
    """将IQ样本转换为通道独立频谱图"""

    def __init__(self):
        pass

    def _normalization(self, data):
        """对输入信号进行归一化处理"""
        s_norm = np.zeros(data.shape, dtype=complex)
        for i in range(data.shape[0]):
            sig_amplitude = np.abs(data[i])
            rms = np.sqrt(np.mean(sig_amplitude**2))
            s_norm[i] = data[i] / rms
        return s_norm

    def _spec_crop(self, x):
        """裁剪生成的通道独立频谱图

        根据频谱图的总行数裁剪掉上下30%的行，只保留中间的40%。
        这种裁剪可能有助于去除频谱图中的冗余信息或噪声
        """
        num_row = x.shape[0]
        x_cropped = x[round(num_row * 0.3) : round(num_row * 0.7)]
        return x_cropped

    def _gen_single_channel_ind_spectrogram(self, sig, win_len=256, overlap=128):
        """
        根据设定的窗口长度和重叠长度，将IQ样本转换为单个通道独立频谱图。

        参数:
        sig: 输入的IQ样本数据。
        win_len (int, 可选): STFT的窗口长度，默认为256。
        overlap (int, 可选): STFT窗口之间的重叠长度，默认为128。

        返回:
        ndarray: 计算得到的通道独立频谱图的幅度对数表示。
        """
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

    def channel_ind_spectrogram(self, data):
        """
        将批量IQ样本转换为通道独立频谱图。

        参数:
        data (ndarray): 输入的IQ样本数据，其中每一行代表一个IQ样本。

        返回:
        ndarray: 一个四维数组，包含所有IQ样本对应的通道独立频谱图。
                形状为 (样本数量, 频谱图行数, 频谱图列数, 1)，其中最后一个维度用于保持数据的一致性。
        """
        # Normalize the IQ samples.
        data = self._normalization(data)

        # Calculate the size of channel independent spectrograms.
        num_sample = data.shape[0]
        num_row = int(256 * 0.4)
        num_column = int(np.floor((data.shape[1] - 256) / 128 + 1) - 1)
        data_channel_ind_spec = np.zeros([num_sample, 1, num_row, num_column])

        # Convert each packet (IQ samples) to a channel independent spectrogram.
        for i in tqdm(range(num_sample)):
            chan_ind_spec_amp = self._gen_single_channel_ind_spectrogram(data[i])
            chan_ind_spec_amp = self._spec_crop(chan_ind_spec_amp)
            data_channel_ind_spec[i, 0, :, :] = chan_ind_spec_amp

        return data_channel_ind_spec

    def wavelet_scattering(self, data, J=6, Q=6):
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
        scattering = Scattering1D(
            J=J,
            Q=Q,
            shape=(sample_length,),
            filter=db1_filter(J=J, shape=data.shape[1]),
        )

        # 计算小波散射输出形状
        scatter_example = scattering(np.real(data[0]))  # 使用第一个样本计算输出
        num_row, num_column = scatter_example.shape  # 确定时间点数和特征数

        # 初始化存储矩阵
        s, e = floor(0.1 * num_row), floor(0.4 * num_row)
        data_wav_spec = np.zeros((num_sample, 2, e - s, num_column - 1))

        for i in tqdm(range(num_sample)):
            sig = np.asarray(data[i])  # 当前样本信号

            # 分离复数信号的实部和虚部
            real_part = np.real(sig)
            imag_part = np.imag(sig)

            # 分别计算实部和虚部的小波散射
            scatter_real = scattering(real_part)
            scatter_imag = scattering(imag_part)

            # 将实部和虚部结果组合（例如计算幅度）
            scatter_combined = np.stack((scatter_real, scatter_imag), axis=0)
            scatter_combined = scatter_combined[:, s:e]
            # Generate channel independent spectrogram.
            scatter_combined = scatter_combined[:, :, 1:] / scatter_combined[:, :, :-1]
            scatter_combined = np.log10(np.abs(scatter_combined) ** 2)

            """绘制结果，不要删除！！！"""
            for i in range(2):
                plt.imshow(
                    scatter_combined[i], aspect="auto", cmap="viridis", origin="lower"
                )
                plt.title("Wavelet Scattering Coefficients")
                plt.xlabel("Time")
                plt.ylabel("Scattering Coefficients")
                plt.colorbar(label="Amplitude")
                plt.show()
                print(1)

            # 存储结果
            data_wav_spec[i] = scatter_combined
        return data_wav_spec
