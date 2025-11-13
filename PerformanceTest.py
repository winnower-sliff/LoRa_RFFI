import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from thop import profile

from core.config import DEVICE


def get_layer_sparsity(layer_stats):
    """从层统计中提取稀疏度"""
    sparsity_dict = {}
    for i, stat in enumerate(layer_stats):
        sparsity_dict[f'layer_{i}'] = stat['sparsity']
    return sparsity_dict


# 模型统计函数
def compute_model_stats(model, model_name="", input_shape=(1, 1, 256)):
    """计算模型的参数量、FLOPs和存储大小"""
    model = model.to(DEVICE)
    stats = {}

    # 计算总参数和非零参数
    total_params = 0
    nonzero_params = 0
    for param in model.parameters():
        if param.requires_grad:
            total_params += param.numel()
            nonzero_params += torch.count_nonzero(param).item()

    stats['total_params'] = total_params
    stats['nonzero_params'] = nonzero_params
    stats['sparsity'] = 1 - (nonzero_params / total_params) if total_params > 0 else 0

    # 计算FLOPs
    try:
        dummy_input = torch.randn(1, *input_shape).to(DEVICE)
        # 对于 TripletNet，只需要一个输入来计算 embedding_net 的 FLOPs
        # 或者使用模型的实际前向传播函数
        if hasattr(model, 'embedding_net'):
            # 如果是 TripletNet，计算 embedding_net 的 FLOPs
            flops, params = profile(model.embedding_net, inputs=(dummy_input,), verbose=False)
        else:
            # 其他模型
            flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        stats['flops'] = flops
    except Exception as e:
        print(f"FLOPs计算失败: {e}")
        stats['flops'] = 0

    # 计算模型存储大小
    try:
        # 保存临时模型文件来计算大小
        temp_path = f"temp_model_{model_name}.pth"
        torch.save(model.state_dict(), temp_path)
        stats['file_size_mb'] = os.path.getsize(temp_path) / (1024 * 1024)
        os.remove(temp_path)
    except Exception as e:
        print(f"模型大小计算失败: {e}")
        stats['file_size_mb'] = 0

    return stats


# 打印模型统计信息
def print_model_stats(original_stats, pruned_stats, model_name=""):
    """打印模型剪枝前后的统计对比"""
    print("\n" + "=" * 60)
    print(f"模型剪枝统计对比 - {model_name}")
    print("=" * 60)

    # 参数量对比
    total_params_reduction = 1 - (pruned_stats['total_params'] / original_stats['total_params'])
    nonzero_params_reduction = 1 - (pruned_stats['nonzero_params'] / original_stats['nonzero_params'])

    print(f"{'指标':<20} {'原始模型':<15} {'剪枝模型':<15} {'减少比例':<15}")
    print("-" * 60)
    print(
        f"{'总参数量':<20} {original_stats['total_params']:<15,} {pruned_stats['total_params']:<15,} {total_params_reduction:.2%}")
    print(
        f"{'非零参数':<20} {original_stats['nonzero_params']:<15,} {pruned_stats['nonzero_params']:<15,} {nonzero_params_reduction:.2%}")
    print(
        f"{'稀疏度':<20} {original_stats['sparsity']:<15.2%} {pruned_stats['sparsity']:<15.2%} +{pruned_stats['sparsity'] - original_stats['sparsity']:.2%}")
    print(
        f"{'FLOPs':<20} {original_stats['flops']:<15,.0f} {pruned_stats['flops']:<15,.0f} {1 - (pruned_stats['flops'] / original_stats['flops']):.2%}")
    print(
        f"{'文件大小(MB)':<20} {original_stats['file_size_mb']:<15.2f} {pruned_stats['file_size_mb']:<15.2f} {1 - (pruned_stats['file_size_mb'] / original_stats['file_size_mb']):.2%}")
    print("=" * 60)


# 可视化统计对比
def visualize_pruning_comparison(original_stats, pruned_stats, model_name="", save_path=None):
    """可视化剪枝前后的统计对比"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # 1. 参数量对比
    labels = ['总参数量', '非零参数']
    original_values = [original_stats['total_params'], original_stats['nonzero_params']]
    pruned_values = [pruned_stats['total_params'], pruned_stats['nonzero_params']]

    x = np.arange(len(labels))
    width = 0.35
    ax1.bar(x - width / 2, original_values, width, label='原始模型', alpha=0.8)
    ax1.bar(x + width / 2, pruned_values, width, label='剪枝模型', alpha=0.8)
    ax1.set_title('参数量对比')
    ax1.set_ylabel('参数数量')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend()

    # 在柱状图上添加数值标签
    for i, v in enumerate(original_values):
        ax1.text(i - width / 2, v, f'{v:,}', ha='center', va='bottom')
    for i, v in enumerate(pruned_values):
        ax1.text(i + width / 2, v, f'{v:,}', ha='center', va='bottom')

    # 2. 性能指标对比
    metrics = ['稀疏度', 'FLOPs', '文件大小(MB)']
    original_metrics = [
        original_stats['sparsity'],
        original_stats['flops'] / 1e6,  # 转换为百万
        original_stats['file_size_mb']
    ]
    pruned_metrics = [
        pruned_stats['sparsity'],
        pruned_stats['flops'] / 1e6,
        pruned_stats['file_size_mb']
    ]

    x = np.arange(len(metrics))
    ax2.bar(x - width / 2, original_metrics, width, label='原始模型', alpha=0.8)
    ax2.bar(x + width / 2, pruned_metrics, width, label='剪枝模型', alpha=0.8)
    ax2.set_title('性能指标对比')
    ax2.set_ylabel('指标值')
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics)
    ax2.legend()

    # 3. 减少比例饼图
    reduction_ratios = [
        1 - pruned_stats['total_params'] / original_stats['total_params'],
        1 - pruned_stats['nonzero_params'] / original_stats['nonzero_params'],
        1 - pruned_stats['flops'] / original_stats['flops'],
        1 - pruned_stats['file_size_mb'] / original_stats['file_size_mb']
    ]
    labels_reduction = ['总参数减少', '非零参数减少', 'FLOPs减少', '文件大小减少']

    ax3.pie(reduction_ratios, labels=labels_reduction, autopct='%1.1f%%', startangle=90)
    ax3.set_title('各项指标减少比例')

    # 4. 稀疏度变化
    layers_original = get_layer_sparsity(original_stats.get('layer_stats', []))
    layers_pruned = get_layer_sparsity(pruned_stats.get('layer_stats', []))

    if layers_original and layers_pruned:
        layer_names = list(layers_original.keys())
        original_sparsity = [layers_original[name] for name in layer_names]
        pruned_sparsity = [layers_pruned[name] for name in layer_names]

        x = range(len(layer_names))
        ax4.plot(x, original_sparsity, 'o-', label='原始模型', linewidth=2)
        ax4.plot(x, pruned_sparsity, 's-', label='剪枝模型', linewidth=2)
        ax4.set_title('各层稀疏度变化')
        ax4.set_xlabel('层索引')
        ax4.set_ylabel('稀疏度')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_xticks(x)
        ax4.set_xticklabels([f'Layer {i}' for i in range(len(layer_names))], rotation=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"统计对比图已保存: {save_path}")

    plt.show()