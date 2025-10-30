"""训练数据准备相关函数"""
import time
import os
import h5py

from training_utils.data_preprocessor import load_data, generate_spectrogram
from utils.signal_trans import awgn


def prepare_train_data(
    new_file_flag,
    filename_train_prepared_data,
    path_train_original_data,
    dev_range,
    pkt_range,
    snr_range,
    generate_type,
    wst_j,
    wst_q
):
    """
    准备训练集

    根据选择的信号处理类型生成对应的训练集
    """
    time_prepare_start = time.time()
    # 数据预处理
    if not os.path.exists(filename_train_prepared_data) or new_file_flag == 1:
        # 需要新处理数据
        print("Data Converting...")

        data, labels = load_data(path_train_original_data, dev_range, pkt_range)
        data = awgn(data, snr_range)

        data = generate_spectrogram(data, generate_type, wst_j, wst_q)

        with h5py.File(filename_train_prepared_data, "w") as f:
            f.create_dataset("data", data=data)
            f.create_dataset("labels", data=labels)
        timeCost = time.time() - time_prepare_start
        print(f"Convert Time Cost: {timeCost:.3f}s")
    else:
        # 处理数据已存在
        print("Data exist, loading...")

        with h5py.File(filename_train_prepared_data, "r") as f:
            data = f["data"][:]
            labels = f["labels"][:]

        timeCost = time.time() - time_prepare_start
        print(f"Load Time Cost: {timeCost:.3f}s")
    return data, labels