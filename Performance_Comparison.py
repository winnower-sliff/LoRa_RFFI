# Performance_Comparison.py
import glob

import torch
import os

from matplotlib import pyplot as plt
from thop import profile
from core.config import DEVICE

from training_utils.data_preprocessor import load_model


def compute_model_stats(model, model_name="", input_shape=(1, 2, 128)):
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
        if hasattr(model, 'embedding_net'):
            flops, params = profile(model.embedding_net, inputs=(dummy_input,), verbose=False)
        else:
            flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        stats['flops'] = flops
    except Exception as e:
        print(f"FLOPs计算失败: {e}")
        stats['flops'] = 0

    # 计算模型存储大小
    try:
        temp_path = f"temp_model_{model_name}.pth"
        torch.save(model.state_dict(), temp_path)
        stats['file_size_mb'] = os.path.getsize(temp_path) / (1024 * 1024)
        os.remove(temp_path)
    except Exception as e:
        print(f"模型大小计算失败: {e}")
        stats['file_size_mb'] = 0

    return stats

def print_model_stats(original_stats, pruned_stats, model_name=""):
    """打印模型剪枝前后的统计对比"""
    print("\n" + "=" * 60)
    print(f"模型剪枝统计对比 - {model_name}")
    print("=" * 60)

    total_params_reduction = 1 - (pruned_stats['total_params'] / original_stats['total_params'])
    nonzero_params_reduction = 1 - (pruned_stats['nonzero_params'] / original_stats['nonzero_params'])

    print(f"{'指标':<20} {'原始模型':<15} {'剪枝模型':<15} {'减少比例':<15}")
    print("-" * 60)
    print(f"{'总参数量':<20} {original_stats['total_params']:<15,} {pruned_stats['total_params']:<15,} {total_params_reduction:.2%}")
    print(f"{'非零参数':<20} {original_stats['nonzero_params']:<15,} {pruned_stats['nonzero_params']:<15,} {nonzero_params_reduction:.2%}")
    print(f"{'稀疏度':<20} {original_stats['sparsity']:<15.2%} {pruned_stats['sparsity']:<15.2%} +{pruned_stats['sparsity'] - original_stats['sparsity']:.2%}")
    print(f"{'FLOPs':<20} {original_stats['flops']:<15,.0f} {pruned_stats['flops']:<15,.0f} {1 - (pruned_stats['flops'] / original_stats['flops']):.2%}")
    print(f"{'文件大小(MB)':<20} {original_stats['file_size_mb']:<15.2f} {pruned_stats['file_size_mb']:<15.2f} {1 - (pruned_stats['file_size_mb'] / original_stats['file_size_mb']):.2%}")
    print("=" * 60)

def print_model_stats_single(model_stats, model_name=""):
    """打印单个模型的统计信息"""
    print("\n" + "=" * 60)
    print(f"模型统计信息 - {model_name}")
    print("=" * 60)
    print(f"{'指标':<20} {'数值':<15}")
    print("-" * 60)
    print(f"{'总参数量':<20} {model_stats['total_params']:<15,}")
    print(f"{'非零参数':<20} {model_stats['nonzero_params']:<15,}")
    print(f"{'稀疏度':<20} {model_stats['sparsity']:<15.2%}")
    print(f"{'FLOPs':<20} {model_stats['flops']:<15,.0f}")
    print(f"{'文件大小(MB)':<20} {model_stats['file_size_mb']:<15.2f}")
    print("=" * 60)

def load_original_model(model_path, net_type, preprocess_type):
    """加载原始模型"""
    return load_model(model_path, net_type, preprocess_type)

def load_pruned_model(model_path, net_type, preprocess_type):
    """加载剪枝模型"""
    return load_model(model_path, net_type, preprocess_type)


def analyze_all_models(model_dir, net_type, preprocess_type):
    """遍历分析指定目录下的所有模型"""
    # 获取原始模型和剪枝模型目录
    origin_dir = os.path.join(model_dir, 'origin')
    prune_dir = os.path.join(model_dir, 'prune')

    # 查找所有模型文件
    origin_models = glob.glob(os.path.join(origin_dir, "Extractor_*.pth"))
    prune_models = glob.glob(os.path.join(prune_dir, "Extractor_*.pth"))

    # 按照epoch排序
    origin_models.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    prune_models.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

    print(f"找到 {len(origin_models)} 个原始模型和 {len(prune_models)} 个剪枝模型")

    # 创建结果存储字典
    results = {}

    # 遍历所有原始模型
    for origin_model_path in origin_models:
        # 提取epoch编号
        epoch = int(origin_model_path.split('_')[-1].split('.')[0])
        pruned_model_path = os.path.join(prune_dir, f"Extractor_{epoch}.pth")

        print(f"\n分析模型 epoch {epoch}...")

        try:
            # 加载原始模型
            original_model = load_original_model(origin_model_path, net_type, preprocess_type)
            original_stats = compute_model_stats(original_model, f"original_epoch_{epoch}")

            # 检查对应的剪枝模型是否存在
            if os.path.exists(pruned_model_path):
                # 加载剪枝模型
                pruned_model = load_pruned_model(pruned_model_path, net_type, preprocess_type)
                pruned_stats = compute_model_stats(pruned_model, f"pruned_epoch_{epoch}")

                # 存储结果
                results[epoch] = {
                    'original': original_stats,
                    'pruned': pruned_stats,
                    'has_pruned': True
                }

                # 打印对比结果
                print_model_stats(original_stats, pruned_stats, f"Epoch {epoch}")
            else:
                # 只有原始模型，没有剪枝模型
                print(f"警告: 找不到对应的剪枝模型 {pruned_model_path}")

                # 存储结果
                results[epoch] = {
                    'original': original_stats,
                    'pruned': None,
                    'has_pruned': False
                }

                # 打印单个模型结果
                print_model_stats_single(original_stats, f"Epoch {epoch} (仅原始模型)")

        except Exception as e:
            print(f"分析模型 epoch {epoch} 时出错: {e}")
            continue

    return results


def visualize_all_models_comparison(results, save_path=None):
    """可视化所有模型的性能对比"""
    # 设置中文字体
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    if not results:
        print("没有数据可以可视化")
        return

    # 准备数据
    epochs = sorted(results.keys())
    original_params = [results[epoch]['original']['total_params'] for epoch in epochs]
    pruned_params = [results[epoch]['pruned']['total_params'] for epoch in epochs]
    original_flops = [results[epoch]['original']['flops'] for epoch in epochs]
    pruned_flops = [results[epoch]['pruned']['flops'] for epoch in epochs]
    original_sparsity = [results[epoch]['original']['sparsity'] for epoch in epochs]
    pruned_sparsity = [results[epoch]['pruned']['sparsity'] for epoch in epochs]
    original_size = [results[epoch]['original']['file_size_mb'] for epoch in epochs]
    pruned_size = [results[epoch]['pruned']['file_size_mb'] for epoch in epochs]

    # 创建图表
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('所有模型性能对比分析', fontsize=16)

    # 1. 参数量趋势
    ax1.plot(epochs, original_params, 'o-', label='原始模型参数量', linewidth=2)
    ax1.plot(epochs, pruned_params, 's-', label='剪枝模型参数量', linewidth=2)
    ax1.set_title('参数量趋势')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('参数数量')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. FLOPs趋势
    ax2.plot(epochs, [f/1e6 for f in original_flops], 'o-', label='原始模型FLOPs', linewidth=2)
    ax2.plot(epochs, [f/1e6 for f in pruned_flops], 's-', label='剪枝模型FLOPs', linewidth=2)
    ax2.set_title('FLOPs趋势')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('FLOPs (百万)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. 稀疏度趋势
    ax3.plot(epochs, original_sparsity, 'o-', label='原始模型稀疏度', linewidth=2)
    ax3.plot(epochs, pruned_sparsity, 's-', label='剪枝模型稀疏度', linewidth=2)
    ax3.set_title('稀疏度趋势')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('稀疏度')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. 文件大小趋势
    ax4.plot(epochs, original_size, 'o-', label='原始模型大小', linewidth=2)
    ax4.plot(epochs, pruned_size, 's-', label='剪枝模型大小', linewidth=2)
    ax4.set_title('模型文件大小趋势')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('文件大小 (MB)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"统计对比图已保存: {save_path}")

    plt.show()

# 使用示例
if __name__ == "__main__":
    # 设置模型目录和参数
    model_dir = "./model/stft/resnet"  # 模型根目录
    net_type = 0  # 根据您的网络类型设置
    preprocess_type = 0  # 根据您的预处理类型设置

    # 遍历分析所有模型
    results = analyze_all_models(model_dir, net_type, preprocess_type)

    # 可选：保存结果到文件
    import json

    with open("model_performance_comparison_results.json", "w") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)
    print("\n结果已保存到 model_performance_comparison_results.json")

    # 可视化所有模型的性能对比
    visualize_all_models_comparison(results, "all_models_comparison.png")

