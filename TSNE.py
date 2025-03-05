import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def tsne_3d_plot(features, labels, perplexity=30, max_iter=1000):
    """
    精简版 3D t-SNE 可视化函数

    参数:
    - features: 输入特征 (Tensor或NumPy数组), 形状 [N, 512]
    - labels: 类别标签 (Tensor或NumPy数组), 形状 [N]
    - ax: 可选的 3D Axes 对象
    - perplexity: t-SNE 困惑度 (默认30)
    - n_iter: 迭代次数 (默认1000)
    """
    # 统一转换为 NumPy 格式
    features = features.numpy() if isinstance(features, torch.Tensor) else features
    labels = labels.numpy() if isinstance(labels, torch.Tensor) else labels

    # 执行 t-SNE 降维（使用 max_iter 代替弃用的 n_iter）
    tsne = TSNE(
        n_components=3, perplexity=perplexity, max_iter=max_iter, random_state=42
    )  # 关键修改点
    embed_3d = tsne.fit_transform(features)

    # 创建 Axes
    plt.switch_backend("qtagg")  # 显式指定交互式后端
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # 绘制 3D 散点图
    scatter = ax.scatter(
        embed_3d[:, 0],
        embed_3d[:, 1],
        embed_3d[:, 2],
        c=labels,
        cmap="tab10",
        alpha=0.6,
        s=10,  # 点大小
    )

    # 添加颜色条和标签
    plt.colorbar(scatter, ax=ax, pad=0.1)
    ax.set_title("3D t-SNE Visualization", pad=15)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.tight_layout()
    plt.show()


# 示例用法
if __name__ == "__main__":

    n_classes = 5
    n_samples_per_class = 20
    n_features = 64

    # 生成随机数据作为示例
    features = torch.randn(n_classes * n_samples_per_class, n_features)
    labels = torch.cat(
        [torch.full((n_samples_per_class,), i) for i in range(n_classes)]
    )

    tsne_3d_plot(features, labels)
