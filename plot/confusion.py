import os

import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrices(wwo_cms, wwo_accs, epoch, net_name, pps_for, vote_size, save_dir):
    """
    绘制不同分类器在有/无投票情况下的混淆矩阵

    :param wwo_cms: 包含混淆矩阵的列表 [无投票矩阵, 有投票矩阵]
    :param wwo_accs: 包含准确率的列表 [无投票准确率, 有投票准确率]
    :param epoch: 当前训练轮数
    :param net_name: 网络名称
    :param pps_for: 预处理类型名称
    :param vote_size: 投票窗口大小
    :param save_dir: 图片保存目录
    """
    fig, axs = plt.subplots(2, 3, figsize=(20, 12))
    # fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    types = ["KNN", "SVM", "Combined"]
    wwo = ["w/o", "w/"]

    for i in range(2):
        for j in range(2 if i == 0 else 3):
            # for j in range(2 if i == 0 else 2):
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
        f"Heatmap Comparison After {epoch} Epochs "
        f"net type: {net_name}, pps: {pps_for}, Vote Size: {vote_size}",
        fontsize=16,
    )

    pic_save_path = save_dir + f"cft_{epoch}.png"
    plt.savefig(pic_save_path)
    print(f"Png save path: {pic_save_path}")
    # plt.show()

def plot_comparison_confusion_matrices(full_model_cms, pruned_model_cms, full_model_accs,
                                     pruned_model_accs, epoch, net_name, pps_for,
                                     vote_size, save_dir, model_names=None):
    """
    绘制完整模型与剪枝模型混淆矩阵的对比图

    :param full_model_cms: 完整模型混淆矩阵 [无投票矩阵, 有投票矩阵]
    :param pruned_model_cms: 剪枝模型混淆矩阵 [无投票矩阵, 有投票矩阵]
    :param full_model_accs: 完整模型准确率 [无投票准确率, 有投票准确率]
    :param pruned_model_accs: 剪枝模型准确率 [无投票准确率, 有投票准确率]
    :param epoch: 当前训练轮数
    :param net_name: 网络名称
    :param pps_for: 预处理类型名称
    :param vote_size: 投票窗口大小
    :param save_dir: 图片保存目录
    :param model_names: 模型名称元组 (默认: ("完整模型", "剪枝模型"))
    """
    if model_names is None:
        model_names = ("完整模型", "剪枝模型")

    fig, axs = plt.subplots(2, 4, figsize=(24, 12))
    types = ["KNN", "SVM", "Combined"]
    wwo = ["无投票", "有投票"]

    # 绘制完整模型结果（左侧）
    for i in range(2):
        for j in range(2 if i == 0 else 3):
            sns.heatmap(
                full_model_cms[i][j],
                annot=True,
                fmt="d",
                cmap="Blues",
                cbar=False,
                square=True,
                ax=axs[i][j],
            )
            axs[i][j].set_title(
                f"{model_names[0]} - {types[j]} {wwo[i]}\n(准确率 = {full_model_accs[i][j] * 100:.2f}%)"
            )
            axs[i][j].set_xlabel("预测标签")
            axs[i][j].set_ylabel("真实标签")

    # 绘制剪枝模型结果（右侧）
    for i in range(2):
        for j in range(2 if i == 0 else 3):
            col_idx = j + 2  # 向右偏移
            sns.heatmap(
                pruned_model_cms[i][j],
                annot=True,
                fmt="d",
                cmap="Blues",
                cbar=False,
                square=True,
                ax=axs[i][col_idx],
            )
            axs[i][col_idx].set_title(
                f"{model_names[1]} - {types[j]} {wwo[i]}\n(准确率 = {pruned_model_accs[i][j] * 100:.2f}%)"
            )
            axs[i][col_idx].set_xlabel("预测标签")
            axs[i][col_idx].set_ylabel("真实标签")

    # 删除空白子图
    fig.delaxes(axs[0, 3])  # 删除第一行第四个子图

    fig.suptitle(
        f"第 {epoch} 轮训练后的模型对比 - "
        f"网络类型: {net_name}, 预处理: {pps_for}, 投票大小: {vote_size}",
        fontsize=16,
    )

    # 创建目录（如果不存在）
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    pic_save_path = os.path.join(save_dir, f"comparison_cft_{epoch}.png")
    plt.savefig(pic_save_path)
    plt.close()  # 关闭图形释放内存
    print(f"对比图保存路径: {pic_save_path}")
