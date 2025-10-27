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
    fig = plt.figure(figsize=(10, 8))
    plt.switch_backend("qtagg")  # 显式指定交互式后端
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

    # 添加颜色条（关联到Figure而非全局plt）
    cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label("Class Label")

    # 设置坐标轴标签和标题（使用ax方法）
    ax.set_title("3D t-SNE Visualization", pad=15)
    ax.set_xlabel("X", labelpad=10)
    ax.set_ylabel("Y", labelpad=10)
    ax.set_zlabel("Z", labelpad=10)

    # 调整布局（操作Figure对象）
    fig.tight_layout()

    return fig  # 返回Figure对象，不自动显示

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

    fig=tsne_3d_plot(features, labels)
    fig.savefig('./tmp.png')
    plt.show()



# def tsne_2d_plot(features, labels, perplexity=30, max_iter=1000, title="2D t-SNE Visualization"):
#     """
#     2D t-SNE 可视化函数
    
#     参数:
#     - features: 输入特征 (Tensor或NumPy数组), 形状 [N, 512]
#     - labels: 类别标签 (Tensor或NumPy数组), 形状 [N]
#     - perplexity: t-SNE 困惑度 (默认30)
#     - max_iter: 最大迭代次数 (默认1000)
#     - title: 图表标题 (默认"2D t-SNE Visualization")
#     """
#     # 统一转换为 NumPy 格式
#     features = features.numpy() if isinstance(features, torch.Tensor) else features
#     labels = labels.numpy() if isinstance(labels, torch.Tensor) else labels
    
#     # 检查标签是否为单值
#     unique_labels = np.unique(labels)
#     n_colors = len(unique_labels)
    
#     # 执行 t-SNE 降维到2D
#     tsne = TSNE(
#         n_components=2,  # 改为2维
#         perplexity=perplexity,
#         max_iter=max_iter,
#         random_state=42
#     )
#     embed_2d = tsne.fit_transform(features)
    
#     # 创建图形
#     plt.figure(figsize=(10, 8))
    
#     # 使用不同的颜色和标记增强区分度
#     colors = plt.cm.tab10(np.linspace(0, 1, n_colors)) if n_colors > 1 else ['blue']
#     markers = ['o', 's', '^', 'v', '<', '>', 'p', '*', 'X', 'D', 'P', 'h', '+', 'x', '|', '_'] * 2
    
#     # 绘制每个类别
#     for i, cls in enumerate(unique_labels):
#         # 选择当前类别的点和标签
#         idx = labels == cls
#         # 使用不同的颜色和标记样式
#         plt.scatter(
#             embed_2d[idx, 0],
#             embed_2d[idx, 1],
#             c=[colors[i]],
#             marker=markers[i],
#             alpha=0.7,
#             s=50,
#             edgecolors='w',
#             linewidth=0.5,
#             label=f'Class {cls}'
#         )
    
#     plt.title(title, fontsize=15, pad=15)
#     plt.xlabel("TSNE Dimension 1", fontsize=12)
#     plt.ylabel("TSNE Dimension 2", fontsize=12)
#     plt.grid(True, linestyle='--', alpha=0.3)
    
#     # 添加图例
#     if len(unique_labels) > 1:
#         plt.legend(loc='best', title='Classes', frameon=True, shadow=True)
    
#     plt.tight_layout()
#     return plt.gcf()  # 返回当前图形对象

# # 示例用法
# if __name__ == "__main__":
#     n_classes = 5
#     n_samples_per_class = 20
#     n_features = 64
    
#     # 生成随机数据作为示例
#     features = torch.randn(n_classes * n_samples_per_class, n_features)
#     labels = torch.cat(
#         [torch.full((n_samples_per_class,), i) for i in range(n_classes)]
#     )
    
#     fig = tsne_2d_plot(features, labels)
#     fig.savefig('./tsne_2d.png', dpi=300, bbox_inches='tight')
#     plt.show()