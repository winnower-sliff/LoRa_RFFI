import argparse
from collections import Counter
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
from net.net_original import *
from net.net_DRSN import *


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    """基础设置"""
    global PROPRECESS_TYPE, TEST_LIST, NET_NAME, PPS_FOR, MODEL_DIR_PATH, WST_J, WST_Q

    # 不要 net_0 & pps_1
    # 0 for origin, 1 for drsn
    net_type = 1
    # 0 for stft, 1 for wst
    PROPRECESS_TYPE = 1
    # "Train" / "Classification" / "Rogue Device Detection"
    mode_type = 0
    # 我们需要一个新的训练文件吗？
    new_file_flag = 1

    TEST_LIST = [1, 5, 10, 20, 50, 100, 150, 200, 250, 300]
    # TEST_LIST = [1]

    WST_J = 7
    WST_Q = 12

    parser = argparse.ArgumentParser(description="参数设置")
    parser.add_argument("-n", "--net", type=int, help="net_type", default=net_type)
    parser.add_argument(
        "-p", "--proprecess", type=int, help="proprecess", default=PROPRECESS_TYPE
    )
    parser.add_argument("-m", "--mode", type=int, help="mode_type", default=mode_type)
    parser.add_argument(
        "-f", "--new_file", type=int, help="NEW_FILE_FLAG", default=new_file_flag
    )
    parser.add_argument("-j", "--wst_j", type=int, help="NEW_FILE_FLAG", default=WST_J)
    parser.add_argument("-q", "--wst_q", type=int, help="NEW_FILE_FLAG", default=WST_Q)

    args = parser.parse_args()
    net_type, PROPRECESS_TYPE, mode_type, new_file_flag, WST_J, WST_Q = (
        args.net,
        args.proprecess,
        args.mode,
        args.new_file,
        args.wst_j,
        args.wst_q,
    )

    print(args)
    # print("Press Enter to continue...")
    # input()  # 等待用户按下Enter键

    """后续设置"""

    if net_type == 0:
        NET_NAME = "origin"
    elif net_type == 1:
        NET_NAME = "drsn"

    if PROPRECESS_TYPE == 0:
        PPS_FOR = "stft"
        MODEL_DIR_PATH = f"./model/{PPS_FOR}/{NET_NAME}/"
        trn_save_data = f"train_data_{PPS_FOR}.h5"
    else:
        PPS_FOR = "wst"

        MODEL_DIR_PATH = f"./model/{PPS_FOR}_j{WST_J}q{WST_Q}/{NET_NAME}/"
        trn_save_data = f"train_data_{PPS_FOR}_j{WST_J}q{WST_Q}.h5"

    if not os.path.exists(MODEL_DIR_PATH):
        os.makedirs(MODEL_DIR_PATH)

    mode_class = ["Train", "Classification", "Rogue Device Detection"]
    RUN_FOR = mode_class[int(mode_type)]

    print(f"Net DIRNAME: {MODEL_DIR_PATH}")

    if RUN_FOR == "Train":
        print(
            R"""
 _____  ____  _        _      ____  ____  _____
/__ __\/  __\/ \  /|  / \__/|/  _ \/  _ \/  __/
  / \  |  \/|| |\ ||  | |\/||| / \|| | \||  \  
  | |  |    /| | \||  | |  ||| \_/|| |_/||  /_ 
  \_/  \_/\_\\_/  \|  \_/  \|\____/\____/\____\
"""
        )

        data_path = "./dataset/Train/dataset_training_aug.h5"
        dev_range = np.arange(0, 30, dtype=int)
        pkt_range = np.arange(0, 1000, dtype=int)
        snr_range = np.arange(20, 80)
        convert_start_time = time.time()

        print(f"Convert Type: {PPS_FOR}")

        if not os.path.exists(trn_save_data) or new_file_flag == 1:
            print("Data Converting...")

            # 加载数据并开始训练
            LoadDatasetObj = LoadDataset()
            data, labels = LoadDatasetObj.load_iq_samples(
                data_path, dev_range, pkt_range
            )
            data = awgn(data, snr_range)
            ChannelIndSpectrogramObj = ChannelIndSpectrogram()

            data = proPrecessData(data, ChannelIndSpectrogramObj)

            with h5py.File(trn_save_data, "w") as f:
                f.create_dataset("data", data=data)
                f.create_dataset("labels", data=labels)
            timeCost = time.time() - convert_start_time
            print(f"Convert Time Cost: {timeCost:.3f}s")
        else:
            print("Data exist, loading...")
            with h5py.File(trn_save_data, "r") as f:
                data = f["data"][:]
                labels = f["labels"][:]

            timeCost = time.time() - convert_start_time
            print(f"Load Time Cost: {timeCost:.3f}s")

        # 训练特征提取模型
        prepare_and_train(data, labels, dev_range, num_epochs=max(TEST_LIST))

    else:
        if RUN_FOR == "Classification":
            print(
                R"""
 ____  _____ _____    _      ____  ____  _____
/   _\/    //__ __\  / \__/|/  _ \/  _ \/  __/
|  /  |  __\  / \    | |\/||| / \|| | \||  \  
|  \__| |     | |    | |  ||| \_/|| |_/||  /_ 
\____/\_/     \_/    \_/  \|\____/\____/\____\
"""
            )

            # 指定设备索引范围用于分类任务
            test_dev_range = np.arange(30, 40, dtype=int)

            # 执行分类任务
            test_classification(
                file_path_enrol="./dataset/Test/dataset_residential.h5",
                file_path_clf="./dataset/Test/channel_problem/A.h5",
                dev_range_enrol=test_dev_range,
                pkt_range_enrol=np.arange(0, 100, dtype=int),
                dev_range_clf=test_dev_range,
                pkt_range_clf=np.arange(100, 200, dtype=int),
            )

        elif RUN_FOR == "Rogue Device Detection":
            print(
                R"""
 ____  ____  ____    _      ____  ____  _____
/  __\/  _ \/  _ \  / \__/|/  _ \/  _ \/  __/
|  \/|| | \|| | \|  | |\/||| / \|| | \||  \  
|    /| |_/|| |_/|  | |  ||| \_/|| |_/||  /_ 
\_/\_\\____/\____/  \_/  \|\____/\____/\____\
"""
            )

            # 执行恶意设备检测任务
            test_rogue_device_detection(
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


# 模型加载函数
def load_model(file_path, weights_only=True):
    """从指定路径加载模型"""
    model = TripletNet(in_channels=1 if PROPRECESS_TYPE == 0 else 2)
    model.load_state_dict(torch.load(file_path, weights_only=weights_only))
    model.eval()
    print(f"Model loaded from {file_path}")
    return model


# 数据预处理函数
def proPrecessData(data, ChannelIndSpectrogramObj: ChannelIndSpectrogram):
    """数据预处理方式选择"""

    if PROPRECESS_TYPE == 0:
        data = ChannelIndSpectrogramObj.channel_ind_spectrogram(data)
    if PROPRECESS_TYPE == 1:
        data = ChannelIndSpectrogramObj.wavelet_scattering(data, J=WST_J, Q=WST_Q)
    return data


# 部分数据加载 & 预处理函数
def load_pps_data(
    file_path,
    dev_range,
    pkt_range,
    LoadDatasetObj: LoadDataset,
    ChannelIndSpectrogramObj: ChannelIndSpectrogram,
):
    """加载 & 预处理数据"""
    data, label = LoadDatasetObj.load_iq_samples(file_path, dev_range, pkt_range)
    data = proPrecessData(data, ChannelIndSpectrogramObj)

    # 准备三元组数据
    triplet_data = [data, data, data]

    # 将三元组输入转换为张量
    triplet_data = [torch.tensor(x).float() for x in triplet_data]

    return label, triplet_data


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

    # 数据集划分
    data_train, data_valid, labels_train, labels_valid = train_test_split(
        data, labels, test_size=0.1, shuffle=True
    )

    # 生成数据加载器
    train_dataset = TripletDataset(data_train, labels_train, dev_range)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    batch_num = math.ceil(len(train_dataset) / batch_size)

    # 初始化模型和优化器
    model = TripletNet(in_channels=1 if PROPRECESS_TYPE == 0 else 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = TripletLoss(margin=0.1)
    # model = load_model(DIR_NAME + f"Extractor_{epoch}.pth")

    # 训练模型
    model.to(DEVICE)
    model.train()

    # 测试用
    # num_epochs = 300

    print(
        "\n---------------------\n"
        "Num of epoch: {}\n"
        "Batch size: {}\n"
        "Num of train batch: {}\n"
        "---------------------\n".format(num_epochs, batch_size, batch_num)
    )
    loss_perepoch = []

    with tqdm(total=num_epochs, desc="Total Progress") as total_bar:
        for epoch in range(num_epochs):
            start_time_ep = time.time()
            total_loss = 0.0
            with tqdm(total=batch_num, desc=f"Epoch {epoch}", leave=False) as pbar:
                for batch_idx, (anchor, positive, negative) in enumerate(train_loader):
                    anchor, positive, negative = (
                        anchor.to(DEVICE),
                        positive.to(DEVICE),
                        negative.to(DEVICE),
                    )

                    # 前向传播
                    embedded_anchor, embedded_positive, embedded_negative = model(
                        anchor, positive, negative
                    )
                    loss = loss_fn(
                        embedded_anchor, embedded_positive, embedded_negative
                    )

                    # 反向传播与优化
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

                    pbar.update(1)

            end_time_ep = time.time()

            loss_ep = total_loss / len(train_loader) * 10

            text = (
                f"Epoch [{epoch+1}/{num_epochs}], "
                + f"time: {end_time_ep-start_time_ep:.2f}s, "
                + f"Loss: {loss_ep:.6f}"
            )

            tqdm.write(text)
            loss_perepoch.append(loss_ep)

            # 保存训练好的模型
            if (epoch + 1) in TEST_LIST:

                # models.append((epoch + 1, copy.deepcopy(model)))
                # print(f"Extractor {epoch+1} saving...")
                # save_model(model=model, file_path=f"./model/wt_1/Extractor_{epoch+1}.pth")

                if not os.path.exists(MODEL_DIR_PATH):
                    os.makedirs(MODEL_DIR_PATH)
                fileName = f"Extractor_{epoch + 1}.pth"
                """保存模型到指定路径"""
                file_path = MODEL_DIR_PATH + fileName
                torch.save(model.state_dict(), file_path)
                tqdm.write(f"Model saved to {file_path}")

                if (epoch + 1) in TEST_LIST[-3:]:
                    # print("Plotting results... ")
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
                    plt.title(
                        f"Loss of {num_epochs} Epoch, Net: {NET_NAME}, Convert Type: {PROPRECESS_TYPE}"
                    )
                    fig.legend(
                        loc="upper right",
                        bbox_to_anchor=(1, 1),
                        bbox_transform=ax1.transAxes,
                    )

                    # 显示图表
                    plt.grid(True)

                    pic_save_path = MODEL_DIR_PATH + f"loss_{epoch+1}.png"
                    plt.savefig(pic_save_path)
                    # plt.show()

            # 更新总进度条
            total_bar.update(1)
    return


# 分类测试函数
def test_classification(
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

    vote_size = 10
    weight_knn = 0.5
    weight_svm = 1 - weight_knn

    """
    提取设备特征
    """

    # 加载注册数据集（IQ样本和标签）
    print("\nData loading...")
    label_enrol, triplet_data_enrol = load_pps_data(
        file_path_enrol,
        dev_range_enrol,
        pkt_range_enrol,
        LoadDatasetObj,
        ChannelIndSpectrogramObj,
    )

    # 加载分类数据集（IQ样本和标签）

    label_clf, triplet_data_clf = load_pps_data(
        file_path_clf,
        dev_range_clf,
        pkt_range_clf,
        LoadDatasetObj,
        ChannelIndSpectrogramObj,
    )
    print("\nData loaded!!!")

    for epoch in TEST_LIST:
        print()
        model = MODEL_DIR_PATH + f"Extractor_{epoch}.pth"
        if os.path.exists(model):
            model = load_model(model)

            # 提取特征
            print("Feature extracting...")
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

            print("Device predicting...")
            clf_start_time = time.time()
            # dev_range_clf = dev_range_clf[dev_range_clf != 39]

            # 提取分类数据集的特征
            with torch.no_grad():
                feature_clf = model(*triplet_data_clf)

            # K-NN和SVM的初步预测
            pred_label_knn_wo = knnclf.predict(feature_clf[0])
            pred_label_svm_wo = svmclf.predict(feature_clf[0])

            # K-NN投票机制
            pred_label_knn_w_v = []
            for i in range(len(pred_label_knn_wo)):
                window_start = max(0, i - vote_size // 2)
                window_end = min(len(pred_label_knn_wo), i + vote_size // 2 + 1)
                window_knn = pred_label_knn_wo[window_start:window_end]
                most_common_label_knn = Counter(window_knn).most_common(1)[0][0]
                pred_label_knn_w_v.append(most_common_label_knn)

            # SVM投票机制
            pred_label_svm_w_v = []
            for i in range(len(pred_label_svm_wo)):
                window_start = max(0, i - vote_size // 2)
                window_end = min(len(pred_label_svm_wo), i + vote_size // 2 + 1)
                window_svm = pred_label_svm_wo[window_start:window_end]
                most_common_label_svm = Counter(window_svm).most_common(1)[0][0]
                pred_label_svm_w_v.append(most_common_label_svm)

            # 综合投票机制
            combined_label = []
            for i in range(0, len(pred_label_knn_w_v), vote_size):
                window_end = min(i + vote_size, len(pred_label_knn_w_v))

                knn_votes = Counter()
                svm_votes = Counter()

                for j in range(i, window_end):
                    knn_votes[pred_label_knn_w_v[j]] += weight_knn
                    svm_votes[pred_label_svm_w_v[j]] += weight_svm
                combined_votes = knn_votes + svm_votes
                final_label = combined_votes.most_common(1)[0][0]

                # 保持与原样本相同的长度
                combined_label.extend([final_label] * (window_end - i))

            # 计算各分类器的准确率
            wo_acc_knn = accuracy_score(label_clf, pred_label_knn_wo)
            wo_acc_svm = accuracy_score(label_clf, pred_label_svm_wo)
            w_acc_knn = accuracy_score(label_clf, pred_label_knn_w_v)
            w_acc_svm = accuracy_score(label_clf, pred_label_svm_w_v)
            acc_combined = accuracy_score(label_clf, combined_label)
            wo_accs = [wo_acc_knn, wo_acc_svm]
            w_accs = [w_acc_knn, w_acc_svm, acc_combined]
            wwo_accs = [wo_accs, w_accs]

            timeCost = time.time() - clf_start_time

            print("-----------------------------")
            print(f"Extractor ID: {epoch}")
            print(f"Vote Size: {vote_size}")
            print(
                f"KNN accuracy\t\tw/o\tvoting = {wo_acc_knn * 100:.2f}%\n"
                f"SVM accuracy\t\tw/o\tvoting = {wo_acc_svm * 100:.2f}%\n"
                f"KNN accuracy\t\tw/\tvoting = {w_acc_knn * 100:.2f}%\n"
                f"SVM accuracy\t\tw/\tvoting = {w_acc_svm * 100:.2f}%\n"
                f"Combined accuracy\tw/\tweighted voting = {acc_combined * 100:.2f}%",
            )
            print(f"Time cost: {timeCost:.3f}s")
            print("-----------------------------")
            print()

            # 绘制混淆矩阵
            conf_mat_knn_wo = confusion_matrix(label_clf, pred_label_knn_w_v)
            conf_mat_svm_wo = confusion_matrix(label_clf, pred_label_svm_w_v)
            conf_mat_knn_w = confusion_matrix(label_clf, pred_label_knn_w_v)
            conf_mat_svm_w = confusion_matrix(label_clf, pred_label_svm_w_v)
            conf_mat_combined = confusion_matrix(label_clf, combined_label)
            wo_cms = [conf_mat_knn_wo, conf_mat_svm_wo]
            w_cms = [conf_mat_knn_w, conf_mat_svm_w, conf_mat_combined]
            wwo_cms = [wo_cms, w_cms]

            fig, axs = plt.subplots(2, 3, figsize=(20, 12))
            types = ["KNN", "SVM", "Combined"]
            wwo = ["w/o", "w/"]

            for i in range(2):
                for j in range(2 if i == 0 else 3):
                    sns.heatmap(
                        wwo_cms[i][j],
                        annot=True,
                        fmt="d",
                        cmap="Blues",
                        cbar=False,
                        square=True,
                        ax=axs[i][j],
                    )
                    axs[i][j].set_title(
                        f"{types[j]} {wwo[i]} Vote (Accuracy = {wwo_accs[i][j] * 100:.2f}%)"
                    )
                    axs[i][j].set_xlabel("Predicted label")
                    axs[i][j].set_ylabel("True label")

            # 删除第一行第三个子图
            fig.delaxes(axs[0, 2])
            fig.suptitle(
                f"Heatmap Comparison After {epoch} Epochs \
                    net type: {NET_NAME}, pps: {PPS_FOR}, Vote Size: {vote_size}, ",
                fontsize=16,
            )

            dir_name = f"{PPS_FOR}_{NET_NAME}_cft/"
            if not os.path.exists(MODEL_DIR_PATH + dir_name):
                os.makedirs(MODEL_DIR_PATH + dir_name)
            pic_save_path = MODEL_DIR_PATH + dir_name + f"cft_{epoch}.png"
            plt.savefig(pic_save_path)
            # plt.show()
        else:
            print(f"Extractor_{epoch}.pth isn't exist")

    return


# 恶意设备检测
def test_rogue_device_detection(
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

    """
    加载注册设备数据
    """
    print("\nDevice enrolling...")
    # 加载注册数据集（IQ样本和标签）
    label_enrol, triplet_data_enrol = load_pps_data(
        file_path_enrol,
        dev_range_enrol,
        pkt_range_enrol,
        LoadDatasetObj,
        ChannelIndSpectrogramObj,
    )

    """
    加载合法设备和恶意设备数据
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
    data_test = proPrecessData(data_test, ChannelIndSpectrogramObj)

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

    for epoch in TEST_LIST:
        print()

        model = load_model(MODEL_DIR_PATH + f"Extractor_{epoch}.pth")

        """
        设备注册
        """

        # 提取特征
        with torch.no_grad():
            feature_enrol = model(*triplet_data_enrol)

        # 构建 K-NN 分类器
        knnclf = KNeighborsClassifier(n_neighbors=15, metric="euclidean")
        knnclf.fit(feature_enrol[0], label_enrol.ravel())

        """
        测试恶意设备检测能力
        """

        # 提取特征
        with torch.no_grad():
            feature_test = model(*triplet_data_test)

        print("Device predicting...")
        # 使用 K-NN 分类器进行预测
        distances, _ = knnclf.kneighbors(feature_test[0])
        detection_score = distances.mean(axis=1)

        # 计算 ROC 曲线和 AUC
        fpr, tpr, thresholds = roc_curve(label_test, detection_score, pos_label=1)
        fpr, tpr = 1 - fpr, 1 - tpr  # 反转 fpr 和 tpr 以匹配距离得分
        eer, eer_threshold = _compute_eer(fpr, tpr, thresholds)
        roc_auc = auc(fpr, tpr)

        print("-----------------------------")
        print(f"Extractor ID: {epoch}")
        print(f"AUC = {roc_auc:.3f}, EER = {eer:.3f}")
        print("-----------------------------")
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

    return


# In[] 主程序执行逻辑
if __name__ == "__main__":
    main()
