from collections import Counter
import copy
import math
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.svm import SVC
import torch
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
from tqdm import tqdm

from signal_trans import *
from net.net_DRSN import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 模型保存函数
def save_model(model, file_path=f"./model/{NET_TYPE}/Extractor.pth"):
    """保存模型到指定路径"""
    torch.save(model.state_dict(), file_path)
    print(f"Model saved to {file_path}")


# 模型加载函数
def load_model(file_path=f"./model/{NET_TYPE}/Extractor.pth", weights_only=True):
    """从指定路径加载模型"""
    model = TripletNet()
    model.load_state_dict(torch.load(file_path, weights_only=weights_only))
    model.eval()
    print(f"Model loaded from {file_path}")
    return model


# 数据准备与模型训练
def prepare_and_train(
    data, labels, dev_range, batch_size=32, num_epochs=200, learning_rate=1e-3
):
    """
    准备数据并训练三元组网络模型。

    参数:
    data: 输入数据，通常为图像特征向量。
    labels: 输入数据的标签。
    dev_range: 设备范围，用于在数据集中选择特定范围的数据进行训练。
    batch_size: 批处理大小，每次迭代训练的网络输入数量。
    num_epochs: 训练的轮数（遍历整个数据集的次数）。
    learning_rate: 学习率，控制优化器更新权重的步长。

    返回:
    models: 训练过程中保存的模型列表，包含每个关键轮次的模型副本。

    过程:
    1. 将数据集划分为训练集和验证集。
    2. 创建三元组数据集和数据加载器。
    3. 初始化三元组网络模型、优化器和损失函数。
    4. 在指定的设备（如GPU）上训练模型。
    5. 在每个epoch结束时，记录损失并保存关键轮次的模型。
    6. 训练结束后，绘制损失随epoch变化的图表。
    """

    models = []

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
    model.to(DEVICE)
    model.train()

    # num_epochs = 10

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
                anchor.to(DEVICE),
                positive.to(DEVICE),
                negative.to(DEVICE),
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

        loss_ep = total_loss / len(train_loader) * 10
        print(
            f"Epoch [{epoch+1}/{num_epochs}],",
            f"time: {end_time_ep-start_time_ep:.2f}s,",
            f"Loss: {loss_ep:.6f}",
        )
        loss_perepoch.append(loss_ep)

        if (epoch + 1) in [1, 5, 10, 20, 50, 100, 150, 200]:

            models.append((epoch + 1, copy.deepcopy(model)))
            # print(f"Extractor {epoch+1} saving...")
            # save_model(model=model, file_path=f"./model/wt_1/Extractor_{epoch+1}.pth")

    # 保存训练好的模型
    if not os.path.exists(f"./model/{NET_TYPE}/"):
        os.makedirs(f"./model/{NET_TYPE}/")
    for ep, model in models:
        save_model(model=model, file_path=f"./model/{NET_TYPE}/Extractor_{ep}.pth")

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

    return models


# 分类测试函数
def test_classification(
    epoch,
    file_path_enrol,
    file_path_clf,
    dev_range_enrol,
    pkt_range_enrol,
    dev_range_clf,
    pkt_range_clf,
):
    """
    使用给定的特征提取模型（从指定路径加载）对注册数据集和分类数据集进行分类测试，
    采用K-NN和SVM分类器进行预测，并返回K-NN分类的预测标签、真实标签以及分类准确率。
    同时，绘制K-NN和SVM分类的混淆矩阵热图。

    参数:
    epoch (int): 用于标识加载的特征提取模型的训练周期数。
    file_path_enrol (str): 注册数据集的文件路径。
    file_path_clf (str): 分类数据集的文件路径。
    dev_range_enrol (tuple): 注册数据集中设备的范围（例如，设备ID的起始和结束值）。
    pkt_range_enrol (tuple): 注册数据集中数据包的范围（例如，数据包的起始和结束索引）。
    dev_range_clf (tuple): 分类数据集中设备的范围。
    pkt_range_clf (tuple): 分类数据集中数据包的范围。

    返回:
    tuple: 包含三个元素的元组，分别是K-NN分类的预测标签（numpy.ndarray）、真实标签（numpy.ndarray）以及K-NN分类准确率（float）。
    """
    # 加载数据
    LoadDatasetObj = LoadDataset()
    ChannelIndSpectrogramObj = ChannelIndSpectrogram()

    model = load_model(f"./model/{NET_TYPE}/Extractor_{epoch}.pth")

    vote_size = 10
    weight_knn = 0.5
    weight_svm = 1 - weight_knn

    """
    提取设备特征
    """

    # 加载注册数据集（IQ样本和标签）
    print("\nData loading...")
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
    ]
    triplet_label_clf = [true_label, true_label, true_label]

    # 将三元组输入转换为张量
    triplet_data_clf = [torch.tensor(x).unsqueeze(1).float() for x in triplet_data_clf]

    # 提取特征
    print("\nFeature extracting...")
    with torch.no_grad():
        feature_enrol = model(*triplet_data_enrol)

    # 使用 K-NN 分类器进行训练
    knnclf = KNeighborsClassifier(n_neighbors=100, metric="euclidean")
    knnclf.fit(feature_enrol[0], label_enrol.ravel())

    svmclf = SVC(kernel="rbf", C=1.0)  # 可以根据需要调整参数
    svmclf.fit(feature_enrol[0], label_enrol.ravel())

    """
    进行预测
    """

    print("\nDevice predicting...")
    clf_start_time = time.time()
    # dev_range_clf = dev_range_clf[dev_range_clf != 39]

    # 提取分类数据集的特征
    with torch.no_grad():
        feature_clf = model(*triplet_data_clf)

    # K-NN和SVM的初步预测
    pred_label_knn = knnclf.predict(feature_clf[0])
    pred_label_svm = svmclf.predict(feature_clf[0])

    # K-NN投票机制
    final_pred_label_knn = []
    for i in range(len(pred_label_knn)):
        window_start = max(0, i - vote_size // 2)
        window_end = min(len(pred_label_knn), i + vote_size // 2 + 1)
        window_knn = pred_label_knn[window_start:window_end]
        most_common_label_knn = Counter(window_knn).most_common(1)[0][0]
        final_pred_label_knn.append(most_common_label_knn)

    # SVM投票机制
    final_pred_label_svm = []
    for i in range(len(pred_label_svm)):
        window_start = max(0, i - vote_size // 2)
        window_end = min(len(pred_label_svm), i + vote_size // 2 + 1)
        window_svm = pred_label_svm[window_start:window_end]
        most_common_label_svm = Counter(window_svm).most_common(1)[0][0]
        final_pred_label_svm.append(most_common_label_svm)

    # 加权投票，结合KNN和SVM的投票结果
    final_combined_label = []
    for i in range(len(final_pred_label_knn)):
        # 根据权重加权统计标签
        knn_vote_weighted = Counter({final_pred_label_knn[i]: weight_knn})
        svm_vote_weighted = Counter({final_pred_label_svm[i]: weight_svm})
        combined_votes = knn_vote_weighted + svm_vote_weighted
        final_label = combined_votes.most_common(1)[0][0]
        final_combined_label.append(final_label)

    # 计算各分类器的准确率
    acc_knn = accuracy_score(true_label, final_pred_label_knn)
    acc_svm = accuracy_score(true_label, final_pred_label_svm)
    acc_combined = accuracy_score(true_label, final_combined_label)
    timeCost = time.time() - clf_start_time

    print()
    print(
        f"KNN accuracy with voting = {acc_knn * 100:.2f}%,",
        f"SVM accuracy with voting = {acc_svm * 100:.2f}%,",
        f"Combined accuracy with weighted voting = {acc_combined * 100:.2f}%",
    )
    # print(
    #     f"KNN voted accuracy = {acc_knn_final * 100:.2f}%,",
    #     f"SVM voted accuracy = {acc_svm_final * 100:.2f}%,",
    # )
    print(
        f"Time cost: {timeCost:.3f}s",
    )
    print()

    # 绘制混淆矩阵
    conf_mat_knn = confusion_matrix(true_label, final_pred_label_knn)
    conf_mat_svm = confusion_matrix(true_label, final_pred_label_svm)
    conf_mat_combined = confusion_matrix(true_label, final_combined_label)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
    sns.heatmap(
        conf_mat_knn, annot=True, fmt="d", cmap="Blues", cbar=False, square=True, ax=ax1
    )
    ax1.set_title(f"KNN with Voting (Accuracy = {acc_knn * 100:.2f}%)")
    ax1.set_xlabel("Predicted label")
    ax1.set_ylabel("True label")

    sns.heatmap(
        conf_mat_svm, annot=True, fmt="d", cmap="Blues", cbar=False, square=True, ax=ax2
    )
    ax2.set_title(f"SVM with Voting (Accuracy = {acc_svm * 100:.2f}%)")
    ax2.set_xlabel("Predicted label")
    ax2.set_ylabel("True label")

    sns.heatmap(
        conf_mat_combined,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        square=True,
        ax=ax3,
    )
    ax3.set_title(f"Combined (Weighted Voting, Accuracy = {acc_combined * 100:.2f}%)")
    ax3.set_xlabel("Predicted label")
    ax3.set_ylabel("True label")

    fig.suptitle(f"Heatmap Comparison After {epoch} Epochs of Training", fontsize=16)
    plt.show()

    return pred_label_knn, true_label, acc_knn


# 恶意设备检测
def test_rogue_device_detection(
    epoch,
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
    该函数使用特征提取模型对恶意设备进行检测，并返回相关的检测结果和性能指标。

    参数:
    epoch (int): 用于加载特征提取模型的训练周期。
    file_path_enrol (str): 注册数据集的路径。
    dev_range_enrol (tuple): 注册数据集中设备的范围。
    pkt_range_enrol (tuple): 注册数据集中数据包的范围。
    file_path_legitimate (str): 合法设备数据集的路径。
    dev_range_legitimate (tuple): 合法设备数据集中设备的范围。
    pkt_range_legitimate (tuple): 合法设备数据集中数据包的范围。
    file_path_rogue (str): 恶意设备数据集的路径。
    dev_range_rogue (tuple): 恶意设备数据集中设备的范围。
    pkt_range_rogue (tuple): 恶意设备数据集中数据包的范围。

    返回:
    fpr (ndarray): 假阳性率。
    tpr (ndarray): 真阳性率。
    roc_auc (float): ROC 曲线下面积。
    eer (float): 等错误率。
    """

    def _compute_eer(fpr, tpr, thresholds):
        """
        计算等错误率 (EER) 和达到 EER 点的阈值。

        参数:
        fpr (ndarray): 假阳性率数组。
        tpr (ndarray): 真阳性率数组。
        thresholds (ndarray): 阈值数组。

        返回:
        eer (float): 等错误率。
        eer_threshold (float): 达到 EER 点的阈值。
        """
        fnr = 1 - tpr
        abs_diffs = np.abs(fpr - fnr)
        min_index = np.argmin(abs_diffs)
        eer = np.mean((fpr[min_index], fnr[min_index]))
        return eer, thresholds[min_index]

    # 加载和处理数据
    LoadDatasetObj = LoadDataset()
    ChannelIndSpectrogramObj = ChannelIndSpectrogram()

    model = load_model(f"./model/{NET_TYPE}/Extractor_{epoch}.pth")

    """
    注册环节
    """
    print("\nDevice enrolling...")
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

    # 构建 K-NN 分类器
    knnclf = KNeighborsClassifier(n_neighbors=15, metric="euclidean")
    knnclf.fit(feature_enrol[0], label_enrol.ravel())

    """
    测试恶意设备检测能力
    """

    print("\nData loading...")
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

    print("\nDevice predicting...")
    # 使用 K-NN 分类器进行预测
    distances, _ = knnclf.kneighbors(feature_test[0])
    detection_score = distances.mean(axis=1)

    # 计算 ROC 曲线和 AUC
    fpr, tpr, thresholds = roc_curve(label_test, detection_score, pos_label=1)
    fpr, tpr = 1 - fpr, 1 - tpr  # 反转 fpr 和 tpr 以匹配距离得分
    eer, eer_threshold = _compute_eer(fpr, tpr, thresholds)
    roc_auc = auc(fpr, tpr)

    print()
    print(f"AUC = {roc_auc:.3f}, EER = {eer:.3f}")
    print()
    # 绘制 ROC 曲线
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}, EER = {eer:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(f"ROC Curve After {epoch} Epochs of Training")
    plt.legend(loc="lower right")
    plt.show()

    return fpr, tpr, roc_auc, eer


# In[] 主程序执行逻辑
if __name__ == "__main__":
    # 指定要运行的任务: "Train" / "Classification" / "Rogue Device Detection"
    mode_class = ["Train", "Classification", "Rogue Device Detection"]
    # mode_type = input(
    #     "0. Train  1. Classification  2. Rogue Device Detection\n模式选择："
    # )
    mode_type = 1
    run_for = mode_class[int(mode_type)]

    if run_for == "Train":
        print("Train mode")

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
        feature_extractors = prepare_and_train(data, labels, dev_range)

    else:
        for ep in [1, 5, 10, 20, 50, 100, 150, 200]:
            if run_for == "Classification":
                print(f"Classification mode, Extractor ID: {ep}")

                # 指定设备索引范围用于分类任务
                test_dev_range = np.arange(30, 40, dtype=int)

                # 执行分类任务
                pred_label, true_label, acc = test_classification(
                    epoch=ep,
                    file_path_enrol="./dataset/Test/dataset_residential.h5",
                    file_path_clf="./dataset/Test/channel_problem/A.h5",
                    dev_range_enrol=test_dev_range,
                    pkt_range_enrol=np.arange(0, 100, dtype=int),
                    dev_range_clf=test_dev_range,
                    pkt_range_clf=np.arange(100, 200, dtype=int),
                )

            elif run_for == "Rogue Device Detection":
                print(f"Rogue Device Detection mode, Extractor ID: {ep}")

                # 执行恶意设备检测任务
                fpr, tpr, roc_auc, eer = test_rogue_device_detection(
                    epoch=ep,
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
