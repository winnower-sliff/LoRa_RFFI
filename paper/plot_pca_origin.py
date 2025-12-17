# paper/plot_pca_origin.py
import os
import sys

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import PreprocessType, NetworkType
from training_utils.data_preprocessor import load_generate_triplet, load_model
from paths import DATASET_FILES, get_model_path, PAPER_OUTPUT_FILES

# ================= 路径设置 =================
PCA_ORIGIN_PLOT_PATH = PAPER_OUTPUT_FILES['pca_origin_pca']

def plot_pca_comparison(feats, n_components=16):
    """
    绘制PCA降维前后的特征对比图 (IEEE单页适配版)
    """
    # 执行PCA
    pca = PCA(n_components=min(n_components, feats.shape[1]))
    pca.fit(feats)

    # 创建对比图 - 调整为单列宽度(约3.5英寸)或双列宽度(约7.5英寸)
    # IEEE单栏宽度约为3.5英寸，双栏宽度约为7.5英寸
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.0))  # 双栏宽度适配

    # 原始特征维度方差
    feature_variances = np.asarray(feats).var(axis=0)
    axes[0].plot(feature_variances, linewidth=1.0)  # 减细线条
    axes[0].set_xlabel('Feature Dimension', fontsize=8)
    axes[0].set_ylabel('Variance', fontsize=8)
    axes[0].tick_params(axis='both', which='major', labelsize=7)
    axes[0].grid(True, alpha=0.3)
    axes[0].text(0.5, -0.25, '(a)', transform=axes[0].transAxes, fontsize=10, fontweight=9, ha='center')

    # PCA后各主成分方差贡献
    explained_variance_ratio = pca.explained_variance_ratio_
    axes[1].bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio,
                width=0.8)  # 调整柱状图宽度
    axes[1].set_xlabel('Principal Component', fontsize=8)
    axes[1].set_ylabel('Explained Variance Ratio', fontsize=8)
    axes[1].tick_params(axis='both', which='major', labelsize=7)
    axes[1].grid(True, alpha=0.3)
    axes[1].text(0.5, -0.25, '(b)', transform=axes[1].transAxes, fontsize=10, fontweight=9, ha='center')
    # 紧凑布局
    plt.tight_layout(pad=0.3)
    plt.savefig(PCA_ORIGIN_PLOT_PATH, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    file_path_enrol = str(DATASET_FILES['test_residential'])
    dev_range_enrol = np.arange(31, 40, dtype=int)
    pkt_range_enrol = np.arange(0, 100, dtype=int)
    model_path = get_model_path('stft', 'LightNet', 'distillation/Extractor_200.pth')
    net_type = NetworkType.LightNet
    preprocess_type = PreprocessType.STFT
    # 加载注册数据集(IQ样本和标签)
    print("\nData loading...")
    label_enrol, triplet_data_enrol = load_generate_triplet(
        file_path_enrol, dev_range_enrol, pkt_range_enrol,
        preprocess_type, snr_range=None
    )

    model = load_model(str(model_path), net_type, preprocess_type)

    with torch.no_grad():
        feature_enrol = model(*triplet_data_enrol)
    plot_pca_comparison(feats=feature_enrol[0], n_components=64)
