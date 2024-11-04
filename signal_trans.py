from numpy import sum, sqrt
from numpy.random import standard_normal, uniform
import scipy.signal as signal
import numpy as np


# Additive White Gaussian Noise (AWGN) Function
def awgn(data, snr_range):
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
        data_channel_ind_spec = np.zeros([num_sample, num_row, num_column, 1])

        # Convert each packet (IQ samples) to a channel independent spectrogram.
        for i in range(num_sample):
            chan_ind_spec_amp = self._gen_single_channel_ind_spectrogram(data[i])
            chan_ind_spec_amp = self._spec_crop(chan_ind_spec_amp)
            data_channel_ind_spec[i, :, :, 0] = chan_ind_spec_amp

        return data_channel_ind_spec
