"""恶意设备检测模式相关函数"""
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier

from experiment_logger import ExperimentLogger
from training_utils.data_preprocessor import load_generate_triplet, load_data, generate_spectrogram, load_model


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
    net_type=None,
    preprocess_type=None,
    test_list=None,
    model_dir_path=None,
    wst_j=None,
    wst_q=None
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
    :param net_type: 网络类型
    :param preprocess_type: 预处理类型
    :param test_list: 测试列表
    :param model_dir_path: 模型目录路径
    :param wst_j: WST J参数
    :param wst_q: WST Q参数

    :return fpr (ndarray): 假阳性率。
    :return tpr (ndarray): 真阳性率。
    :return roc_auc (float): ROC 曲线下面积。
    :return eer (float): 等错误率。
    """

    # 初始化实验记录
    logger = ExperimentLogger()
    exp_config = {
        "mode": "rogue_device_detection",
        "model": {
            "type": net_type
        },
        "data": {
            "preprocess_type": preprocess_type,
            "test_points": test_list,
            "enrol_file": file_path_enrol,
            "legitimate_file": file_path_legitimate,
            "rogue_file": file_path_rogue
        }
    }
    exp_filepath, exp_id = logger.create_experiment_record(exp_config)

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
        file_path_enrol, dev_range_enrol, pkt_range_enrol, preprocess_type
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
    data_test = generate_spectrogram(data_test, preprocess_type, wst_j, wst_q)

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

    detection_results = {}

    for epoch in test_list or []:
        print()
        model_path = model_dir_path + f"Extractor_{epoch}.pth"
        model = load_model(model_path, net_type, preprocess_type)

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

        detection_results[f"epoch_{epoch}"] = {
            "auc": float(roc_auc),
            "eer": float(eer),
            "eer_threshold": float(eer_threshold)
        }

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

    # 记录实验结果
    final_results = {
        "rogue_detection": detection_results,
        "model_dir_path": model_dir_path
    }
    logger.update_experiment_result(exp_id, final_results)

    return