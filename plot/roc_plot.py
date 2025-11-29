import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc


def compute_eer(fpr, tpr, thresholds):
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

def evaluate_and_plot_roc(label_test, detection_score, epoch):
        """
        计算ROC曲线指标并绘制ROC曲线

        :param label_test: 测试标签
        :param detection_score: 检测分数
        :param epoch: 当前轮次
        :return: fpr, tpr, roc_auc, eer, eer_threshold
        """
        # 计算 ROC 曲线和 AUC
        fpr, tpr, thresholds = roc_curve(label_test, detection_score, pos_label=1)
        fpr, tpr = 1 - fpr, 1 - tpr  # 反转 fpr 和 tpr 以匹配距离得分
        eer, eer_threshold = compute_eer(fpr, tpr, thresholds)
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

        return fpr, tpr, roc_auc, eer, eer_threshold
