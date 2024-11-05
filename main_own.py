import math
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

import torch
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
from tqdm import tqdm

from signal_trans import *
from net import *
from utils import *


# 数据准备与模型训练
# 调整后的模型准备和训练调用
def prepare_and_train(
    data, labels, dev_range, batch_size=32, num_epochs=200, learning_rate=1e-3
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据集划分
    data_train, data_valid, labels_train, labels_valid = train_test_split(
        data, labels, test_size=0.1, shuffle=True
    )

    # 生成数据加载器
    train_dataset = TripletDataset(data_train, labels_train, dev_range)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    batch_num = math.ceil(len(train_dataset) / batch_size)

    # 初始化模型和优化器
    model = TripletNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = TripletLoss(margin=0.1)

    # 训练模型
    model.to(device)
    model.train()

    print(
        "\n-----------------\n"
        "Num of epoch: {}\n"
        "Batch size: {}\n"
        "Num of train batch: {}\n"
        "-----------------\n".format(num_epochs, batch_size, batch_num)
    )
    loss_perepoch = []

    for epoch in range(num_epochs):
        start_time_ep = time.time()
        total_loss = 0.0
        pbar = tqdm(
            enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}"
        )
        for batch_idx, (anchor, positive, negative) in pbar:
            anchor, positive, negative = (
                anchor.to(device),
                positive.to(device),
                negative.to(device),
            )

            # 前向传播
            embedded_anchor, embedded_positive, embedded_negative = model(
                anchor, positive, negative
            )
            loss = loss_fn(embedded_anchor, embedded_positive, embedded_negative)

            # 反向传播与优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        end_time_ep = time.time()


        loss_ep=total_loss/len(train_loader)*10
        print(
            f"Epoch [{epoch+1}/{num_epochs}],",
            f"time: {end_time_ep-start_time_ep:.2f}s,",
            f"Loss: {loss_ep:.6f}",
        )
        loss_perepoch.append(loss_ep)

        if (epoch + 1) in [100, 150, 200]:
            print(f"Extractor {epoch+1} saving...")
            save_model(model=model, file_path=f"Extractor_{epoch+1}.pth")

    print("Plotting results... ")
    fig, ax1 = plt.subplots()
    ax1.plot(
        range(len(loss_perepoch)),
        loss_perepoch,
        label="Loss",
        color="red",
    )
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="red")
    ax1.tick_params(axis="y", labelcolor="red")

    # 添加标题和图例
    plt.title("Loss of Each Epoch")
    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)

    # 显示图表
    plt.grid(True)
    plt.show()

    return model


# 分类测试函数
def test_classification(
    model,
    file_path_enrol,
    file_path_clf,
    dev_range_enrol,
    pkt_range_enrol,
    dev_range_clf,
    pkt_range_clf,
):
    """
    使用特征提取模型进行分类测试，并返回分类准确率。
    """
    # 加载数据
    LoadDatasetObj = LoadDataset()
    ChannelIndSpectrogramObj = ChannelIndSpectrogram()

    # 加载注册数据集（IQ样本和标签）
    data_enrol, label_enrol = LoadDatasetObj.load_iq_samples(
        file_path_enrol, dev_range_enrol, pkt_range_enrol
    )
    data_enrol = ChannelIndSpectrogramObj.channel_ind_spectrogram(data_enrol).squeeze(3)

    # 准备三元组数据
    triplet_data_enrol = [
        data_enrol,
        data_enrol,
        data_enrol,
    ]
    triplet_label_enrol = [label_enrol, label_enrol, label_enrol]

    # 将三元组输入转换为张量
    triplet_data_enrol = [
        torch.tensor(x).unsqueeze(1).float() for x in triplet_data_enrol
    ]

    # 提取特征
    with torch.no_grad():
        feature_enrol = model(*triplet_data_enrol)

    # 使用 K-NN 分类器进行训练
    knnclf = KNeighborsClassifier(n_neighbors=15, metric="euclidean")
    knnclf.fit(feature_enrol[0], label_enrol.ravel())

    # 加载分类数据集（IQ样本和标签）
    data_clf, true_label = LoadDatasetObj.load_iq_samples(
        file_path_clf, dev_range_clf, pkt_range_clf
    )
    data_clf = ChannelIndSpectrogramObj.channel_ind_spectrogram(data_clf).squeeze(3)

    # 准备三元组数据
    triplet_data_clf = [
        data_clf,
        data_clf,
        data_clf,
    ]  # 这里简单地将数据复制为三元组形式
    triplet_label_clf = [true_label, true_label, true_label]

    # 将三元组输入转换为张量
    triplet_data_clf = [torch.tensor(x).unsqueeze(1).float() for x in triplet_data_clf]

    # 提取分类数据集的特征
    with torch.no_grad():
        feature_clf = model(*triplet_data_clf)

    # 进行预测
    pred_label = knnclf.predict(feature_clf[0])
    acc = accuracy_score(true_label, pred_label)
    print(f"Overall accuracy = {acc:.4f}")

    # 绘制混淆矩阵
    conf_mat = confusion_matrix(true_label, pred_label)
    plt.figure()
    sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Epoch 200")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.show()

    return pred_label, true_label, acc


# 恶意设备检测
def test_rogue_device_detection(
    model,
    file_path_enrol,
    dev_range_enrol,
    pkt_range_enrol,
    file_path_legitimate,
    dev_range_legitimate,
    pkt_range_legitimate,
    file_path_rogue,
    dev_range_rogue,
    pkt_range_rogue,
):
    """
    使用特征提取模型进行恶意设备检测，并返回检测结果。
    """

    def _compute_eer(fpr, tpr, thresholds):
        """计算等错误率 (EER) 和达到 EER 点的阈值。"""
        fnr = 1 - tpr
        abs_diffs = np.abs(fpr - fnr)
        min_index = np.argmin(abs_diffs)
        eer = np.mean((fpr[min_index], fnr[min_index]))
        return eer, thresholds[min_index]

    # 加载和处理数据
    LoadDatasetObj = LoadDataset()
    ChannelIndSpectrogramObj = ChannelIndSpectrogram()

    data_enrol, label_enrol = LoadDatasetObj.load_iq_samples(
        file_path_enrol, dev_range_enrol, pkt_range_enrol
    )
    data_enrol = ChannelIndSpectrogramObj.channel_ind_spectrogram(data_enrol).squeeze(3)

    # 准备三元组数据
    triplet_data_enrol = [
        data_enrol,
        data_enrol,
        data_enrol,
    ]
    triplet_label_enrol = [label_enrol, label_enrol, label_enrol]

    # 将三元组输入转换为张量
    triplet_data_enrol = [
        torch.tensor(x).unsqueeze(1).float() for x in triplet_data_enrol
    ]

    # 提取特征
    with torch.no_grad():
        feature_enrol = model(*triplet_data_enrol)

    # 构建 K-NN 分类器
    knnclf = KNeighborsClassifier(n_neighbors=15, metric="euclidean")
    knnclf.fit(feature_enrol[0], label_enrol.ravel())

    # 加载合法设备和恶意设备数据
    data_legitimate, _ = LoadDatasetObj.load_iq_samples(
        file_path_legitimate, dev_range_legitimate, pkt_range_legitimate
    )
    data_rogue, _ = LoadDatasetObj.load_iq_samples(
        file_path_rogue, dev_range_rogue, pkt_range_rogue
    )

    # 合并合法设备和恶意设备数据
    data_test = np.concatenate([data_legitimate, data_rogue])
    label_test = np.concatenate(
        [np.ones(len(data_legitimate)), np.zeros(len(data_rogue))]
    )

    # 提取特征
    data_test = ChannelIndSpectrogramObj.channel_ind_spectrogram(data_test).squeeze(3)

    # 准备三元组数据
    triplet_data_test = [
        data_test,
        data_test,
        data_test,
    ]
    triplet_label_test = [label_test, label_test, label_test]

    # 将三元组输入转换为张量
    triplet_data_test = [
        torch.tensor(x).unsqueeze(1).float() for x in triplet_data_test
    ]

    # 提取特征
    with torch.no_grad():
        feature_test = model(*triplet_data_test)

    # 使用 K-NN 分类器进行预测
    distances, _ = knnclf.kneighbors(feature_test[0])
    detection_score = distances.mean(axis=1)

    # 计算 ROC 曲线和 AUC
    fpr, tpr, thresholds = roc_curve(label_test, detection_score, pos_label=1)
    fpr, tpr = 1 - fpr, 1 - tpr  # 反转 fpr 和 tpr 以匹配距离得分
    eer, eer_threshold = _compute_eer(fpr, tpr, thresholds)
    roc_auc = auc(fpr, tpr)

    # 绘制 ROC 曲线
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}, EER = {eer:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC Curve    Ex 200")
    plt.legend(loc="lower right")
    plt.show()

    return fpr, tpr, roc_auc, eer


# In[] 主程序执行逻辑
if __name__ == "__main__":
    # 指定要运行的任务: "Train" / "Classification" / "Rogue Device Detection"
    mode_class = ["Train", "Classification", "Rogue Device Detection"]
    mode_type = 2
    run_for = mode_class[mode_type]

    if run_for == "Train":
        print("Train mode entering...")

        data_path = "./dataset/Train/dataset_training_aug.h5"
        save_data = "./train_data.h5"
        dev_range = np.arange(0, 30, dtype=int)
        pkt_range = np.arange(0, 1000, dtype=int)
        snr_range = np.arange(20, 80)
        if not os.path.exists(save_data):
            print("Data Converting...")

            # 加载数据并开始训练
            LoadDatasetObj = LoadDataset()
            data, labels = LoadDatasetObj.load_iq_samples(
                data_path, dev_range, pkt_range
            )
            data = awgn(data, snr_range)
            ChannelIndSpectrogramObj = ChannelIndSpectrogram()
            data = ChannelIndSpectrogramObj.channel_ind_spectrogram(data)

            data = np.squeeze(data, axis=3)

            with h5py.File(save_data, "w") as f:
                f.create_dataset("data", data=data)
                f.create_dataset("labels", data=labels)
        else:
            print("Data loading...")
            with h5py.File(save_data, "r") as f:
                data = f["data"][:]
                labels = f["labels"][:]

        # 训练特征提取模型
        feature_extractor = prepare_and_train(data, labels, dev_range)

        # 保存训练好的模型
        save_model(feature_extractor)

    elif run_for == "Classification":
        print("Classification mode entering...")

        # 指定设备索引范围用于分类任务
        test_dev_range = np.arange(30, 40, dtype=int)

        # 执行分类任务
        model = load_model("Extractor_200.pth")
        pred_label, true_label, acc = test_classification(
            model,
            file_path_enrol="./dataset/Test/dataset_residential.h5",
            file_path_clf="./dataset/Test/channel_problem/A.h5",
            dev_range_enrol=test_dev_range,
            pkt_range_enrol=np.arange(0, 100, dtype=int),
            dev_range_clf=test_dev_range,
            pkt_range_clf=np.arange(100, 200, dtype=int),
        )

    elif run_for == "Rogue Device Detection":
        print("Rogue Device Detection mode entering...")

        # 执行恶意设备检测任务
        model = load_model("Extractor_200.pth")
        fpr, tpr, roc_auc, eer = test_rogue_device_detection(
            model,
            file_path_enrol="./dataset/Test/dataset_residential.h5",
            dev_range_enrol=np.arange(30, 40, dtype=int),
            pkt_range_enrol=np.arange(0, 100, dtype=int),
            file_path_legitimate="./dataset/Test/dataset_residential.h5",
            dev_range_legitimate=np.arange(30, 40, dtype=int),
            pkt_range_legitimate=np.arange(100, 200, dtype=int),
            file_path_rogue="./dataset/Test/dataset_rogue.h5",
            dev_range_rogue=np.arange(40, 45, dtype=int),
            pkt_range_rogue=np.arange(0, 100, dtype=int),
        )
