import argparse
from collections import Counter
import math
import time
import h5py
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

from TripletDataset import TripletDataset, TripletLoss
from net.TripletNet import TripletNet
from signal_trans import awgn

from TSNE import tsne_3d_plot
from utils import generate_spectrogram, load_data, load_generate_triplet, load_model
from better_print import TextAnimator, print_colored_text


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    """基础设置"""
    global NET_TYPE, PROPRECESS_TYPE, TEST_LIST, NET_NAME, PPS_FOR, MODEL_DIR_PATH, WST_J, WST_Q

    # 不要 net_0 & pps_1
    # 0 for origin, 1 for drsn
    NET_TYPE = 1
    # 0 for stft, 1 for wst
    PROPRECESS_TYPE = 0
    # "Train" / "Classification" / "Rogue Device Detection"
    mode_type = 0
    # 我们需要一个新的训练文件吗？
    new_file_flag = 1

    TEST_LIST = [1, 5, 10, 20, 50, 100, 150, 200, 250, 300]
    # TEST_LIST = [1]

    WST_J = 6
    WST_Q = 6

    parser = argparse.ArgumentParser(description="参数设置")
    parser.add_argument("-n", "--net", type=int, help="net_type", default=NET_TYPE)
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
    NET_TYPE, PROPRECESS_TYPE, mode_type, new_file_flag, WST_J, WST_Q = (
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

    if NET_TYPE == 0:
        NET_NAME = "origin"
    elif NET_TYPE == 1:
        NET_NAME = "drsn"

    if PROPRECESS_TYPE == 0:
        PPS_FOR = "stft"
        MODEL_DIR_PATH = f"./model/{PPS_FOR}/{NET_NAME}/"
        filename_train_prepared_data = f"train_data_{PPS_FOR}.h5"
    else:
        PPS_FOR = "wst"
        MODEL_DIR_PATH = f"./model/{PPS_FOR}_j{WST_J}q{WST_Q}/{NET_NAME}/"
        filename_train_prepared_data = f"train_data_{PPS_FOR}_j{WST_J}q{WST_Q}.h5"

    if not os.path.exists(MODEL_DIR_PATH):
        os.makedirs(MODEL_DIR_PATH)

    mode_class = ["Train", "Classification", "Rogue Device Detection"]
    RUN_FOR = mode_class[int(mode_type)]

    print(f"Net DIRNAME: {MODEL_DIR_PATH}")

    if RUN_FOR == "Train":
        # 训练模式
        print_colored_text("训练模式", "32")

        print(f"Convert Type: {PPS_FOR}")

        data, labels = prepare_train_data(
            new_file_flag,
            filename_train_prepared_data,
            path_train_original_data="4307data/DATA_shen_tx1-40_pktN800_433m_1M_10gain.h5",
            dev_range=np.arange(0, 40, dtype=int),
            pkt_range=np.arange(0, 800, dtype=int),
            snr_range=np.arange(20, 80),
            generate_type=PROPRECESS_TYPE,
        )

        # 训练特征提取模型
        train(data, labels, num_epochs=max(TEST_LIST))

    elif RUN_FOR == "Classification":
        print_colored_text("分类模式", "32")

        # 执行分类任务
        test_classification(
            file_path_enrol="4307data/3.13tmp/DATA_all_dev_1~11_300times_433m_1M_3gain.h5",
            file_path_clf="4307data/3.13tmp/DATA_lab2_dev_8_8_3_3_7_7_5_5_500times_433m_500k_70gain.h5",
            dev_range_enrol=np.arange(0, 11, dtype=int),
            pkt_range_enrol=np.arange(0, 300, dtype=int),
            dev_range_clf=np.array([81,82,31,32,71,72,51,52])-1,
            pkt_range_clf=np.arange(0, 500, dtype=int),
        )

    elif RUN_FOR == "Rogue Device Detection":
        print_colored_text("甄别恶意模式", "32")

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


def prepare_train_data(
    new_file_flag,
    filename_train_prepared_data,
    path_train_original_data,
    dev_range,
    pkt_range,
    snr_range,
    generate_type,
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

        data = generate_spectrogram(data, generate_type, WST_J, WST_Q)

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


# 数据准备与模型训练
def train(data, labels, batch_size=32, num_epochs=200, learning_rate=1e-3):
    """
    准备数据并训练三元组网络模型。

    过程:
    1. 将数据集划分为训练集和验证集(尽管此函数只使用了训练集)。
    2. 创建三元组数据集(TripletDataset)和数据加载器(DataLoader)。
    3. 初始化三元组网络模型(TripletNet)、优化器(如Adam)和损失函数(如TripletLoss)。
    4. 将模型移动到指定的设备(如GPU)上, 并设置模型为训练模式。
    5. 在每个epoch中, 遍历数据加载器, 进行前向传播、计算损失、反向传播和优化步骤。
    6. 记录每个epoch的损失, 并在指定的轮次(TEST_LIST)保存模型状态字典。
    7. 在训练的最后几个轮次(TEST_LIST[-3:]), 绘制损失随epoch变化的图表并保存。

    :param data: 输入数据, 通常为图像特征向量。
    :param labels: 输入数据的标签。
    :param batch_size (int): 批处理大小, 每次迭代训练的网络输入数量。默认为32。
    :param num_epochs (int): 训练的轮数(遍历整个数据集的次数)。默认为200。
    :param learning_rate (float): 学习率, 控制优化器更新权重的步长。默认为1e-3。
    """

    # 数据集划分
    data_train, data_valid, labels_train, labels_valid = train_test_split(
        data, labels, test_size=0.1, shuffle=True
    )

    # 生成数据加载器
    train_dataset = TripletDataset(data_train, labels_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    batch_num = math.ceil(len(train_dataset) / batch_size)

    # 初始化模型和优化器
    model = TripletNet(net_type=NET_TYPE, in_channels=1 if PROPRECESS_TYPE == 0 else 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = TripletLoss(margin=0.1)

    # 训练模型
    model.to(DEVICE)
    model.train()

    print(
        "\n---------------------\n"
        "Num of epoch: {}\n"
        "Batch size: {}\n"
        "Num of train batch: {}\n"
        "---------------------\n".format(num_epochs, batch_size, batch_num)
    )
    loss_per_epoch = []

    # 总进度条
    with tqdm(total=num_epochs, desc="Total Progress") as total_bar:
        for epoch in range(num_epochs):
            start_time_ep = time.time()
            total_loss = 0.0
            # 每一轮训练进度条
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
            loss_per_epoch.append(loss_ep)

            # 保存训练好的模型
            if (epoch + 1) in TEST_LIST:
                # 创建文件夹&文件
                if not os.path.exists(MODEL_DIR_PATH):
                    os.makedirs(MODEL_DIR_PATH)
                file_name = f"Extractor_{epoch + 1}.pth"

                # 保存模型到指定路径
                file_path = MODEL_DIR_PATH + file_name
                torch.save(model.state_dict(), file_path)
                tqdm.write(f"Model saved to {file_path}")

                # 绘制loss折线图
                if (epoch + 1) in TEST_LIST[-3:]:
                    # print("Plotting results... ")
                    fig, ax1 = plt.subplots()
                    ax1.plot(
                        range(len(loss_per_epoch)),
                        loss_per_epoch,
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
    dev_range_enrol: list[int] = None,
    dev_range_clf: list[int] = None,
    pkt_range_enrol: list[int] = None,
    pkt_range_clf: list[int] = None,
) -> None:
    """
    * 使用给定的特征提取模型(从指定路径加载)对注册数据集和分类数据集进行分类测试。

    :param file_path_enrol (str): 注册数据集的文件路径。
    :param file_path_clf (str): 分类数据集的文件路径。
    :param dev_range_enrol (tuple): 注册数据集中设备的范围(例如, 设备ID的起始和结束值)。
    :param pkt_range_enrol (tuple): 注册数据集中数据包的范围(例如, 数据包的起始和结束索引)。
    :param dev_range_clf (tuple): 分类数据集中设备的范围。
    :param pkt_range_clf (tuple): 分类数据集中数据包的范围。
    """
    # 加载数据

    vote_size = 10
    weight_knn = 0.5
    weight_svm = 1 - weight_knn

    """
    提取设备特征
    """

    # 加载注册数据集(IQ样本和标签)
    print("\nData loading...")
    label_enrol, triplet_data_enrol = load_generate_triplet(
        file_path_enrol, dev_range_enrol, pkt_range_enrol, PROPRECESS_TYPE
    )

    # 加载分类数据集(IQ样本和标签)

    label_clf, triplet_data_clf = load_generate_triplet(
        file_path_clf, dev_range_clf, pkt_range_clf, PROPRECESS_TYPE
    )
    print("\nData loaded!!!")

    for epoch in TEST_LIST:
        print()
        print("=============================")
        model = MODEL_DIR_PATH + f"Extractor_{epoch}.pth"
        if not os.path.exists(model):
            print(f"Extractor_{epoch}.pth isn't exist")
        else:
            model = load_model(model, NET_TYPE, PROPRECESS_TYPE)
            print("Model loaded!!!")

            # 提取特征
            # print("Feature extracting...")
            try:
                text = TextAnimator("Feature extracting", "Feature extracted")
                text.start()
                with torch.no_grad():
                    feature_enrol = model(*triplet_data_enrol)

                # 使用 K-NN 分类器进行训练
                knnclf = KNeighborsClassifier(n_neighbors=100, metric="euclidean")
                knnclf.fit(feature_enrol[0], label_enrol.ravel())

                svmclf = SVC(kernel="rbf", C=1.0)  # 可以根据需要调整参数
                svmclf.fit(feature_enrol[0], label_enrol.ravel())
            finally:
                text.stop()

            """
            进行预测
            """

            # print("Device predicting...")
            try:
                text = TextAnimator("Device predicting", "Device prediction finish")
                text.start()

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

            finally:
                text.stop()

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
            print("-----------------------------")
            print()

            # 绘制混淆矩阵
            conf_mat_knn_wo = confusion_matrix(label_clf, pred_label_knn_wo)
            conf_mat_svm_wo = confusion_matrix(label_clf, pred_label_svm_wo)
            conf_mat_knn_w = confusion_matrix(label_clf, pred_label_knn_w_v)
            conf_mat_svm_w = confusion_matrix(label_clf, pred_label_svm_w_v)
            conf_mat_combined = confusion_matrix(label_clf, combined_label)
            wo_cms = [conf_mat_knn_wo, conf_mat_svm_wo]
            w_cms = [conf_mat_knn_w, conf_mat_svm_w, conf_mat_combined]
            wwo_cms = [wo_cms, w_cms]

            # fig, axs = plt.subplots(2, 3, figsize=(20, 12))
            fig, axs = plt.subplots(2, 2, figsize=(12, 12))
            types = ["KNN", "SVM", "Combined"]
            wwo = ["w/o", "w/"]

            for i in range(2):
                # for j in range(2 if i == 0 else 3):
                for j in range(2 if i == 0 else 2):
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
            # fig.delaxes(axs[0, 2])
            fig.suptitle(
                f"Heatmap Comparison After {epoch} Epochs \
                    net type: {NET_NAME}, pps: {PPS_FOR}, Vote Size: {vote_size}",
                fontsize=16,
            )

            dir_name = f"{PPS_FOR}_{NET_NAME}_cft/"
            if not os.path.exists(MODEL_DIR_PATH + dir_name):
                os.makedirs(MODEL_DIR_PATH + dir_name)
            pic_save_path = MODEL_DIR_PATH + dir_name + f"cft_{epoch}.png"
            plt.savefig(pic_save_path)
            print(f"Png save path: {pic_save_path}")
            # plt.show()

            # T-SNE 3D绘图
            # tsne_3d_plot(feature_clf[0],labels=label_clf)
        print("=============================")

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
    该函数使用特征提取模型对恶意设备进行检测, 并返回相关的检测结果和性能指标。

    :param file_path_enrol (str): 注册数据集的路径。
    :param dev_range_enrol (tuple): 注册数据集中设备的范围。
    :param pkt_range_enrol (tuple): 注册数据集中数据包的范围。
    :param file_path_legitimate (str): 合法设备数据集的路径。
    :param dev_range_legitimate (tuple): 合法设备数据集中设备的范围。
    :param pkt_range_legitimate (tuple): 合法设备数据集中数据包的范围。
    :param file_path_rogue (str): 恶意设备数据集的路径。
    :param dev_range_rogue (tuple): 恶意设备数据集中设备的范围。
    :param pkt_range_rogue (tuple): 恶意设备数据集中数据包的范围。

    :return fpr (ndarray): 假阳性率。
    :return tpr (ndarray): 真阳性率。
    :return roc_auc (float): ROC 曲线下面积。
    :return eer (float): 等错误率。
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
    """
    加载注册设备数据
    """
    print("\nDevice enrolling...")
    # 加载注册数据集(IQ样本和标签)
    label_enrol, triplet_data_enrol = load_generate_triplet(
        file_path_enrol, dev_range_enrol, pkt_range_enrol, PROPRECESS_TYPE
    )

    """
    加载合法设备和恶意设备数据
    """

    print("\nData loading...")
    # 加载合法设备和恶意设备数据
    data_legitimate, _ = load_data(
        file_path_legitimate, dev_range_legitimate, pkt_range_legitimate
    )
    data_rogue, _ = load_data(file_path_rogue, dev_range_rogue, pkt_range_rogue)

    # 合并合法设备和恶意设备数据
    data_test = np.concatenate([data_legitimate, data_rogue])
    label_test = np.concatenate(
        [np.ones(len(data_legitimate)), np.zeros(len(data_rogue))]
    )

    # 提取特征
    data_test = generate_spectrogram(data_test, PROPRECESS_TYPE, WST_J, WST_Q)

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
        model = MODEL_DIR_PATH + f"Extractor_{epoch}.pth"
        model = load_model(model, NET_TYPE, PROPRECESS_TYPE)

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
