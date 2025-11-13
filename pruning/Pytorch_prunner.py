"""剪枝模式相关函数 - 集成PyTorch原生剪枝"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from PerformanceTest import compute_model_stats
from net.TripletNet import TripletNet


def pytorch_native_prune(model: nn.Module, pruning_rates, method='ln', show_model_summary=False):
    """
    应用通道级剪枝
    """

    # 获取所有可剪枝的卷积层和全连接层
    prunable_layers = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            # 跳过分类层和某些关键层
            if 'classification' in name:
                continue
            prunable_layers.append((name, module))

    if show_model_summary:
        print(f"找到 {len(prunable_layers)} 个可剪枝层")

    # 确保剪枝率数量匹配
    if len(pruning_rates) != len(prunable_layers):
        print(f"警告: 剪枝率数量({len(pruning_rates)})与可剪枝层数量({len(prunable_layers)})不匹配")
        if len(pruning_rates) < len(prunable_layers):
            pruning_rates = list(pruning_rates) + [0.0] * (len(prunable_layers) - len(pruning_rates))
        else:
            pruning_rates = pruning_rates[:len(prunable_layers)]

    # 应用剪枝
    for i, (name, module) in enumerate(prunable_layers):
        pruning_rate = pruning_rates[i]

        if pruning_rate <= 0:
            if show_model_summary:
                print(f"跳过层 {name}, 剪枝率: {pruning_rate}")
            continue

        if show_model_summary:
            print(f"剪枝层 {name}, 剪枝率: {pruning_rate:.2%}")

        if method == 'l1':
            # 基于L1范数的非结构化剪枝
            prune.l1_unstructured(module, name='weight', amount=pruning_rate)
        elif method == 'random':
            # 随机剪枝
            prune.random_unstructured(module, name='weight', amount=pruning_rate)
        elif method == 'ln':
            # 基于L2范数的结构化剪枝
            prune.ln_structured(module, name='weight', amount=pruning_rate, n=2, dim=0)
        else:
            prune.l1_unstructured(module, name='weight', amount=pruning_rate)

    return model


def compute_pruning_rates(model: nn.Module, target_sparsity=0.5, method='uniform', show_model_summary=False):
    """
    计算各层的剪枝率

    Args:
        model: 目标模型
        target_sparsity: 目标总体稀疏度
        method: 剪枝率分配方法 ('uniform', 'global')
    """

    prunable_layers = []
    pruning_rates = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            if 'classification' in name:
                continue
            prunable_layers.append((name, module))

    if method == 'uniform':
        # 均匀剪枝：所有层使用相同的剪枝率
        return [target_sparsity] * len(prunable_layers)

    elif method == 'global':
        # 全局剪枝：基于全局重要性排序
        all_weights = []
        layer_params = []

        # 计算每层权重的重要性
        for name, module in prunable_layers:
            weight = module.weight.data
            if weight.dim() == 4:  # Conv2d
                # 计算每个输出通道的L1范数
                channel_importance = weight.abs().sum(dim=(1, 2, 3))
            elif weight.dim() == 2:  # Linear
                # 计算每个输出神经元的L1范数
                channel_importance = weight.abs().sum(dim=1)
            else:
                channel_importance = weight.abs().sum()

            all_weights.extend(channel_importance.cpu().numpy())
            layer_params.append(len(channel_importance))

        # 全局排序并计算阈值
        all_weights = np.array(all_weights)
        threshold = np.percentile(all_weights, target_sparsity * 100)

        # 计算各层剪枝率
        start_idx = 0
        for num_params in layer_params:
            layer_weights = all_weights[start_idx:start_idx + num_params]
            pruned_count = np.sum(layer_weights < threshold)
            pruning_rate = pruned_count / num_params if num_params > 0 else 0
            pruning_rates.append(pruning_rate)
            start_idx += num_params

    elif method == 'tinyml_gradient':
        # TinyML梯度自适应剪枝：基于权重重要性排名的梯度变化确定剪枝率
        # 首先需要获取每层的权重重要性排名

        # 计算每层权重的重要性排名
        layer_ranks = []
        for name, module in prunable_layers:
            weight = module.weight.data
            if weight.dim() == 4:  # Conv2d
                # 计算每个输出通道的L1范数并排序
                channel_importance = weight.abs().sum(dim=(1, 2, 3))
                sorted_indices = torch.argsort(channel_importance, descending=True)
                ranks = torch.argsort(sorted_indices).cpu().numpy()  # 排名从0开始
            elif weight.dim() == 2:  # Linear
                # 计算每个输出神经元的L1范数并排序
                neuron_importance = weight.abs().sum(dim=1)
                sorted_indices = torch.argsort(neuron_importance, descending=True)
                ranks = torch.argsort(sorted_indices).cpu().numpy()
            else:
                # 其他情况使用整体L1范数
                importance = weight.abs().sum()
                ranks = np.array([0])

            layer_ranks.append(ranks)

        # 应用梯度计算方法（模仿 gratitude_pr 函数）
        n = 5  # 梯度窗口大小，可以根据需要调整

        for rank in layer_ranks:
            # 计算梯度
            gra = []
            for idx, r in enumerate(rank):
                if idx == len(rank) - n:
                    break
                try:
                    g = (rank[idx + n] - r) / n  # 计算梯度
                    gra.append(g)
                except IndexError:
                    pass

            # 寻找最大梯度点并计算剪枝率
            if len(gra) > 0:
                max_gradient_idx = np.argmax(gra)
                idx_position = int(max_gradient_idx + n / 2)
                rate = 1 - (idx_position / len(gra)) if len(gra) > 0 else target_sparsity
                # 限制剪枝率范围
                rate = np.clip(rate, 0.0, 0.9)
            else:
                rate = target_sparsity

            pruning_rates.append(rate)

        # 确保剪枝率数量与可剪枝层数量一致
        if len(pruning_rates) != len(prunable_layers):
            if len(pruning_rates) < len(prunable_layers):
                pruning_rates = list(pruning_rates) + [target_sparsity] * (len(prunable_layers) - len(pruning_rates))
            else:
                pruning_rates = pruning_rates[:len(prunable_layers)]

    if show_model_summary:
        print(f"使用剪枝方法: {method}")
        print(f"目标稀疏度: {target_sparsity:.2%}")
        print(
            f"剪枝率统计: 平均 {np.mean(pruning_rates):.2%}, 最大 {np.max(pruning_rates):.2%}, 最小 {np.min(pruning_rates):.2%}")
        for i, rate in enumerate(pruning_rates):
            print(f"  层 {i}: {rate:.2%}")

    return pruning_rates


def remove_pruning_masks_and_apply(model: nn.Module):
    """
    移除剪枝掩码并使剪枝永久化，返回清理后的状态字典
    """
    # 创建原始状态字典的副本
    original_state_dict = model.state_dict()
    cleaned_state_dict = {}

    # 处理剪枝后的参数
    for name, param in original_state_dict.items():
        if name.endswith('_orig'):
            # 这是剪枝后的原始权重，对应的mask也在状态字典中
            base_name = name[:-5]  # 移除 '_orig'
            mask_name = base_name + '_mask'

            if mask_name in original_state_dict:
                # 应用mask得到最终权重
                mask = original_state_dict[mask_name]
                final_weight = param * mask
                cleaned_state_dict[base_name] = final_weight
            else:
                cleaned_state_dict[base_name] = param
        elif name.endswith('_mask'):
            # 跳过mask，因为我们已经在上面的步骤中处理了
            continue
        else:
            # 普通参数，直接复制
            cleaned_state_dict[name] = param

    # 重新加载清理后的状态字典
    model.load_state_dict(cleaned_state_dict, strict=False)

    # 使用PyTorch的remove_prune来清理剪枝结构
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            if hasattr(module, 'weight_mask'):
                try:
                    prune.remove(module, 'weight')
                except:
                    pass  # 如果已经移除，忽略错误

    return model


def get_clean_state_dict(model: nn.Module):
    """
    获取清理后的状态字典（移除剪枝相关的临时参数）
    """
    original_state_dict = model.state_dict()
    cleaned_state_dict = {}

    for name, param in original_state_dict.items():
        if not name.endswith('_orig') and not name.endswith('_mask'):
            cleaned_state_dict[name] = param

    return cleaned_state_dict


def pytorch_prune_model(pruned_embedding_net, net_type, preprocess_type, show_model_summary=False):

    # 获取清理后的状态字典
    cleaned_state_dict = get_clean_state_dict(pruned_embedding_net)

    # 创建新的TripletNet（不使用剪枝文件）
    pruned_triplet_net = TripletNet(
        net_type=net_type,  # 使用原始网络类型，不是剪枝类型
        in_channels=1 if preprocess_type == 0 else 2,
        custom_pruning_file=None,
    )

    # 加载清理后的权重
    missing_keys, unexpected_keys = pruned_triplet_net.embedding_net.load_state_dict(
        cleaned_state_dict, strict=False
    )

    if missing_keys:
        print(f"警告: 缺失的键: {missing_keys}")
    if unexpected_keys:
        print(f"警告: 意外的键: {unexpected_keys}")

    # 计算剪枝模型统计
    try:
        print("\n计算剪枝模型统计...")
        original_stats = compute_model_stats(
            pruned_triplet_net,
            "original",
            input_shape=(1 if preprocess_type == 0 else 2, 256, 256)
        )

        print("\n剪枝模型统计信息:")
        print(f"参数总数: {original_stats.get('total_params', 'N/A')}")
        print(f"FLOPs: {original_stats.get('flops', 'N/A')}")
        print(f"稀疏度: {original_stats.get('sparsity', 'N/A')}")
        print(f"内存占用: {original_stats.get('memory_usage', 'N/A')}")

    except Exception as e:
        print(f"模型统计计算失败: {e}")
        print("继续执行，不影响模型创建...")

    # 清理可能存在的thop hooks
    for module in pruned_triplet_net.modules():
        if hasattr(module, '_forward_hooks'):
            module._forward_hooks.clear()
        if hasattr(module, 'total_ops'):
            delattr(module, 'total_ops')
        if hasattr(module, 'total_params'):
            delattr(module, 'total_params')

    return pruned_triplet_net