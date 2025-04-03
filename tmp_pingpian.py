from random import randint, random
from matplotlib import pyplot as plt
from numpy.random import standard_normal, uniform
import h5py
import numpy as np

from LoadDataset import LoadDataset


dataset_name = "data"
labelset_name = "label"

file_path = "./4307data/data_2025.2.17.h5"

dev_range, pkt_range = np.arange(0, 6, dtype=int), np.arange(000, 200, dtype=int)
calibration = False


with h5py.File(file_path, "r") as f:
    # LoadDatasetobj = LoadDataset()
    data, labels = LoadDataset.load_iq_samples(file_path, dev_range)
    num_pkt = data.shape[0] // dev_range.shape[0]
    # 计算频偏
    F_dream = 433.125 * 10**6

    F_Rx_wo_err = F_dream
    F_Rx_err = randint(0, int(F_Rx_wo_err * 0.01))

    fs = 1000 * 10**3
    N = 8192  # 信号长度

    devs_cents = []
    for devN in range(len(dev_range)):
        cents = []
        for i in range(num_pkt):
            signal = data[num_pkt * devN + i]
            # 计算FFT
            fft_result = np.fft.fft(signal)
            frequencies = np.fft.fftfreq(N, 1 / fs)

            # 计算频谱质心
            magnitude = np.abs(fft_result)
            f_centroid = np.sum(frequencies * magnitude) / np.sum(magnitude)
            cents.append(f_centroid)

            print(f"Device [{devN}], 频谱质心估计的中心频率: {f_centroid/1e3:.2f} kHz")

            # 绘制频谱
            if 1 == 0:
                plt.title(f"Frequency Offset: {f_centroid/1e3:.2f} kHz + Rx_Error kHz")
                plt.plot(frequencies, 20 * np.log10(np.abs(fft_result)))
                plt.axvline(
                    f_centroid, color="g", linestyle="--", label="Estimated CFO"
                )
                plt.xlabel("Frequency (Hz)")
                plt.ylabel("Magnitude (dB)")
                plt.legend()
                plt.show()
        devs_cents.append(cents)


    # 绘制频谱
    if 1 == 1:
        # 创建一个颜色列表，确保每个设备有不同的颜色
        colors = plt.cm.get_cmap("tab10", len(dev_range))  # 使用 'tab10' 色图获取 n 种颜色

        # 绘制折线图
        for i in range(len(dev_range)):
            plt.plot(devs_cents[i][:], label=f"Device [{i}]", color=colors(i))

        # 添加图例
        plt.legend()

        # 添加标题和标签
        plt.title("Data for Multiple Devices")
        plt.xlabel("Data Point Index")
        plt.ylabel("Data Value")

        # 显示图形
        plt.show()
