
import yaml
import matplotlib.pyplot as plt
import numpy as np


def extract_accuracy_data(results, min_epoch=50):
    """从分类结果中提取准确率数据"""
    epochs = []
    knn_wo_voting = []
    svm_wo_voting = []
    knn_w_voting = []
    svm_w_voting = []
    combined_w_weighted_voting = []

    # 按照epoch顺序排序
    sorted_epochs = sorted(results.keys(), key=lambda x: int(x.replace('epoch', '')))

    for epoch_key in sorted_epochs:
        epoch_num = int(epoch_key.replace('epoch', ''))
        # 过滤掉小于min_epoch的epoch
        if epoch_num < min_epoch:
            continue

        epoch_data = results[epoch_key]
        epochs.append(int(epoch_key.replace('epoch', '')))
        knn_wo_voting.append(epoch_data['accuracies']['knn_wo_voting'])
        svm_wo_voting.append(epoch_data['accuracies']['svm_wo_voting'])
        knn_w_voting.append(epoch_data['accuracies']['knn_w_voting'])
        svm_w_voting.append(epoch_data['accuracies']['svm_w_voting'])
        combined_w_weighted_voting.append(epoch_data['accuracies']['combined_w_weighted_voting'])

    return epochs, knn_wo_voting, svm_wo_voting, knn_w_voting, svm_w_voting, combined_w_weighted_voting


def plot_classification_comparison(origin_data, pruned_data, metric_name, save_path=None):
    """绘制分类结果对比图"""
    origin_epochs, origin_values = origin_data
    pruned_epochs, pruned_values = pruned_data

    plt.figure(figsize=(10, 6))
    plt.plot(origin_epochs, origin_values, 'o-', label=f'原始模型 - {metric_name}', linewidth=2)
    plt.plot(pruned_epochs, pruned_values, 's-', label=f'剪枝模型 - {metric_name}', linewidth=2)

    plt.xlabel('训练轮次 (Epoch)')
    plt.ylabel('准确率')
    plt.title(f'{metric_name} 对比图')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 设置x轴刻度
    all_epochs = sorted(set(origin_epochs + pruned_epochs))
    plt.xticks(all_epochs)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_comprehensive_comparison(origin_results, pruned_results, save_path=None):
    """绘制综合对比图"""
    # 设置中文字体
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 提取数据
    origin_epochs, origin_knn_wo, origin_svm_wo, origin_knn_w, origin_svm_w, origin_combined = extract_accuracy_data(origin_results)
    pruned_epochs, pruned_knn_wo, pruned_svm_wo, pruned_knn_w, pruned_svm_w, pruned_combined = extract_accuracy_data(pruned_results)

    plt.figure(figsize=(12, 8))

    # 原始模型
    plt.plot(origin_epochs, origin_knn_w, 'o--', label='原始模型 - KNN有投票', linewidth=1.5, alpha=0.8)
    plt.plot(origin_epochs, origin_svm_w, 's--', label='原始模型 - SVM有投票', linewidth=1.5, alpha=0.8)
    plt.plot(origin_epochs, origin_combined, '^-', label='原始模型 - 综合加权投票', linewidth=2)

    # 剪枝模型
    plt.plot(pruned_epochs, pruned_knn_w, 'o-', label='剪枝模型 - KNN有投票', linewidth=1.5, alpha=0.8)
    plt.plot(pruned_epochs, pruned_svm_w, 's-', label='剪枝模型 - SVM有投票', linewidth=1.5, alpha=0.8)
    plt.plot(pruned_epochs, pruned_combined, 'v-', label='剪枝模型 - 综合加权投票', linewidth=2)

    plt.xlabel('训练轮次 (Epoch)')
    plt.ylabel('准确率')
    plt.title('分类结果对比图')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)

    # 设置x轴刻度
    all_epochs = sorted(set(origin_epochs + pruned_epochs))
    plt.xticks(all_epochs)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_separate_comparison_subplots(origin_results, pruned_results, save_path=None):
    """绘制拆分子图的对比图"""
    # 设置中文字体
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 提取数据
    origin_epochs, origin_knn_wo, origin_svm_wo, origin_knn_w, origin_svm_w, origin_combined = extract_accuracy_data(origin_results)
    pruned_epochs, pruned_knn_wo, pruned_svm_wo, pruned_knn_w, pruned_svm_w, pruned_combined = extract_accuracy_data(pruned_results)

    # 创建包含6个子图的图表
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('原始模型 vs 剪枝模型分类结果对比', fontsize=16)

    # 子图1: KNN无投票
    axes[0, 0].plot(origin_epochs, origin_knn_wo, 'o--', label='原始模型', linewidth=1.5)
    axes[0, 0].plot(pruned_epochs, pruned_knn_wo, 's-', label='剪枝模型', linewidth=1.5)
    axes[0, 0].set_title('KNN无投票')
    axes[0, 0].set_xlabel('训练轮次 (Epoch)')
    axes[0, 0].set_ylabel('准确率')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 子图2: SVM无投票
    axes[0, 1].plot(origin_epochs, origin_svm_wo, 'o--', label='原始模型', linewidth=1.5)
    axes[0, 1].plot(pruned_epochs, pruned_svm_wo, 's-', label='剪枝模型', linewidth=1.5)
    axes[0, 1].set_title('SVM无投票')
    axes[0, 1].set_xlabel('训练轮次 (Epoch)')
    axes[0, 1].set_ylabel('准确率')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 子图3: KNN有投票
    axes[1, 0].plot(origin_epochs, origin_knn_w, 'o--', label='原始模型', linewidth=1.5)
    axes[1, 0].plot(pruned_epochs, pruned_knn_w, 's-', label='剪枝模型', linewidth=1.5)
    axes[1, 0].set_title('KNN有投票')
    axes[1, 0].set_xlabel('训练轮次 (Epoch)')
    axes[1, 0].set_ylabel('准确率')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 子图4: SVM有投票
    axes[1, 1].plot(origin_epochs, origin_svm_w, 'o--', label='原始模型', linewidth=1.5)
    axes[1, 1].plot(pruned_epochs, pruned_svm_w, 's-', label='剪枝模型', linewidth=1.5)
    axes[1, 1].set_title('SVM有投票')
    axes[1, 1].set_xlabel('训练轮次 (Epoch)')
    axes[1, 1].set_ylabel('准确率')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # 子图5: 综合加权投票
    axes[2, 0].plot(origin_epochs, origin_combined, 'o--', label='原始模型', linewidth=1.5)
    axes[2, 0].plot(pruned_epochs, pruned_combined, 's-', label='剪枝模型', linewidth=1.5)
    axes[2, 0].set_title('综合加权投票')
    axes[2, 0].set_xlabel('训练轮次 (Epoch)')
    axes[2, 0].set_ylabel('准确率')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)

    # 隐藏第六个子图
    axes[2, 1].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    MODEL_DIR = f"./model/stft/resnet/performance_records.yaml"
    SAVE_DIR = f"./model/stft/resnet/comprehensive_classification_comparison.png"

    # 读取YAML数据
    with open(MODEL_DIR, 'r') as f:
        data = yaml.safe_load(f)

    # 获取原始模型和剪枝模型的分类结果
    origin_results = data['models']['origin']['classification_history']
    pruned_results = data['models']['pruned']['classification_history']

    # 绘制综合对比图
    # plot_comprehensive_comparison(origin_results, pruned_results, SAVE_DIR)

    # 绘制拆分子图的对比图
    plot_separate_comparison_subplots(origin_results, pruned_results, SAVE_DIR)