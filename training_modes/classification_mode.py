"""分类模式相关函数"""
import os
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
from tqdm import tqdm

from training_utils.data_preprocessor import load_generate_triplet, load_model
from utils.better_print import TextAnimator


def test_classification(
    file_path_enrol,
    file_path_clf,
    dev_range_enrol=None,
    dev_range_clf=None,
    pkt_range_enrol=None,
    pkt_range_clf=None,
    net_type=None,
    preprocess_type=None,
    test_list=None,
    model_dir_path=None,
    wst_j=None,
    wst_q=None,
    net_name=None,
    pps_for=None
):
    """
    * 使用给定的特征提取模型(从指定路径加载)对注册数据集和分类数据集进行分类测试。

    :param file_path_enrol (str): 注册数据集的文件路径。
    :param file_path_clf (str): 分类数据集的文件路径。
    :param dev_range_enrol: 注册数据集中设备的范围。
    :param dev_range_clf: 分类数据集中设备的范围。
    :param pkt_range_enrol: 注册数据集中数据包的范围。
    :param pkt_range_clf: 分类数据集中数据包的范围。
    :param net_type: 网络类型
    :param preprocess_type: 预处理类型
    :param test_list: 测试点列表
    :param model_dir_path: 模型目录路径
    :param wst_j: WST J参数
    :param wst_q: WST Q参数
    :param net_name: 网络名称
    :param pps_for: 预处理类型名称
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
        file_path_enrol, dev_range_enrol, pkt_range_enrol, preprocess_type
    )

    # 加载分类数据集(IQ样本和标签)

    label_clf, triplet_data_clf = load_generate_triplet(
        file_path_clf, dev_range_clf, pkt_range_clf, preprocess_type
    )
    print("\nData loaded!!!")

    for epoch in test_list or []:
        print()
        print("=============================")
        model_path = model_dir_path + f"Extractor_{epoch}.pth"
        if not os.path.exists(model_path):
            print(f"Extractor_{epoch}.pth isn't exist")
        else:
            model = load_model(model_path, net_type, preprocess_type)
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
                f"Heatmap Comparison After {epoch} Epochs "
                f"net type: {net_name}, pps: {pps_for}, Vote Size: {vote_size}",
                fontsize=16,
            )

            dir_name = f"{pps_for}_{net_name}_cft/"
            if not os.path.exists(model_dir_path + dir_name):
                os.makedirs(model_dir_path + dir_name)
            pic_save_path = model_dir_path + dir_name + f"cft_{epoch}.png"
            plt.savefig(pic_save_path)
            print(f"Png save path: {pic_save_path}")
            # plt.show()

            # T-SNE 3D绘图
            # tsne_3d_plot(feature_clf[0],labels=label_clf)
        print("=============================")

    return