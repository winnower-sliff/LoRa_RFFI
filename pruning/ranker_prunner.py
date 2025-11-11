"""
Author: Ryan
结合 rank_gen.py 和 automatic_prunner.py
用于计时整个剪枝过程，而不是rank_gen时间 + automatic_pruner时间

"""
import math
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
# from thop import profile  # 用于计算FLOPs
from torch.utils.data import DataLoader
from tqdm import tqdm

# 从配置模块导入
from core.config import DEVICE
from core.config import NetworkType, CUSTOM_PRUNING_FILE, PRUNED_OUTPUT_DIR
from net.TripletNet import TripletNet
from net.net_prune import pruned_drsnet18
from training_utils.TripletDataset import TripletDataset, TripletLoss
from training_utils.data_preprocessor import load_model


def mapping(W, min_w, max_w):
    """权重映射函数"""
    scale_w = (max_w - min_w) / 100
    min_arr = torch.full(W.shape, min_w, device=W.device)
    q_w = torch.round((W - min_arr) / scale_w).to(torch.uint8)
    return q_w

def extract_weight(model: nn.Module, prune_rank_path: str, prune_ranks_path: str, show_model_summary=False):
    """提取PyTorch模型的权重"""
    results = []
    idx_results = []

    print("开始提取模型权重...")

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            # 跳过分类层
            if 'classification' in name:
                continue

            weight = module.weight.data
            if show_model_summary:
                print(f"处理层: {name}, 权重形状: {weight.shape}")

            if weight.ndim == 4:  # Conv2D层
                # 重塑权重为 (filters, -1)
                a = weight.view(weight.size(0), -1)
            elif weight.ndim == 2:  # Linear层
                # 重塑权重为 (units, -1)
                a = weight
            elif weight.ndim == 3:  # Conv1D层
                # 重塑权重为 (filters, -1)
                a = weight.view(weight.size(0), -1)
            else:
                continue

            n_filters = a.shape[0]
            r = torch.norm(a, dim=1)
            r = mapping(r, torch.min(r), torch.max(r))
            results.append(sorted(r.cpu().numpy(), reverse=True))
            idx_dis = torch.argsort(r, dim=0)
            idx_results.append(idx_dis.cpu().numpy())

    # 保存结果
    df = pd.DataFrame(results, index=None)
    df.to_csv(prune_rank_path, header=False, index=False)
    df = pd.DataFrame(idx_results, index=None)
    df.to_csv(prune_ranks_path, header=False, index=False)

    print(f"权重提取完成! 生成文件:")
    print(f"  - {prune_rank_path}")
    print(f"  - {prune_ranks_path}")

    return results, idx_results

def fpgm(model: nn.Module, output_dir: Path, dist_type="l2"):
    """PyTorch版本的FPGM剪枝"""
    results = []
    idx_results = []

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            print(f"处理层: {name}")
            weight = module.weight.data

            if weight.ndim == 4:  # Conv2D
                weight_vec = weight.view(weight.size(0), -1).t()
            elif weight.ndim == 3:  # Conv1D
                weight_vec = weight.view(weight.size(0), -1).t()
            elif weight.ndim == 2:  # Linear
                weight_vec = weight.t()
            else:
                continue

            # 计算距离矩阵
            if dist_type in ["l2", "l1"]:
                dist_matrix = torch.cdist(weight_vec, weight_vec, p=2)
            elif dist_type == "cos":
                norm_weight = F.normalize(weight_vec, p=2, dim=1)
                dist_matrix = 1 - torch.mm(norm_weight, norm_weight.t())

            squeeze_matrix = torch.sum(torch.abs(dist_matrix), dim=0)
            distance_sum = sorted(squeeze_matrix.cpu().numpy(), reverse=True)
            idx_dis = torch.argsort(squeeze_matrix, dim=0)
            r = mapping(torch.tensor(distance_sum),
                           torch.min(torch.tensor(distance_sum)),
                           torch.max(torch.tensor(distance_sum)))

            results.append(r.cpu().numpy())
            idx_results.append(idx_dis.cpu().numpy())

    # 保存结果
    output_dir.mkdir(exist_ok=True)
    df = pd.DataFrame(results, index=None)
    df.to_csv(output_dir.joinpath("fpgm.csv"), header=False)
    df = pd.DataFrame(idx_results, index=None)
    df.to_csv(output_dir.joinpath("fpgm_idx.csv"), header=False)

    return results, idx_results

# 新增：打印模型统计信息
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

# 新增：可视化统计对比
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

def get_layer_sparsity(layer_stats):
    """从层统计中提取稀疏度"""
    sparsity_dict = {}
    for i, stat in enumerate(layer_stats):
        sparsity_dict[f'layer_{i}'] = stat['sparsity']
    return sparsity_dict

# 新增：模型统计函数
def compute_model_stats(model, model_name="", input_shape=(1, 1, 256)):
    """计算模型的参数量、FLOPs和存储大小"""
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

def copy_weights(original_model: nn.Module, pruned_model: nn.Module, ranks_path: str, show_model_summary=False):
    """复制权重到剪枝后的模型"""

    def show_print(*args):
        if show_model_summary:
            print(args)

    # 获取状态字典
    original_state_dict = original_model.state_dict()
    pruned_state_dict = pruned_model.state_dict()

    # 打印模型结构信息用于调试
    if show_model_summary:
        print("\n原始模型结构:")
        for name, param in original_model.named_parameters():
            print(f"  {name}: {param.shape}")

        print("\n剪枝模型结构:")
        for name, param in pruned_model.named_parameters():
            print(f"  {name}: {param.shape}")

    # 权重复制逻辑
    new_state_dict = {}

    print("\n开始复制模型权重...")
    for key in pruned_state_dict.keys():
        if key in original_state_dict:
            orig_weight = original_state_dict[key]
            pruned_shape = pruned_state_dict[key].shape

            if orig_weight.shape == pruned_shape:
                # 形状相同，直接复制
                new_state_dict[key] = orig_weight.clone()
                show_print(f"直接复制: {key}")
            else:
                # 形状不同，需要选择性复制
                show_print(f"形状不匹配 - {key}: {orig_weight.shape} -> {pruned_shape}")

                # 根据不同的层类型处理权重复制
                if 'conv' in key and 'weight' in key:
                    # 卷积层权重处理
                    if len(orig_weight.shape) == 4:  # Conv2d权重
                        out_c, in_c, h, w = pruned_shape
                        orig_out_c, orig_in_c, orig_h, orig_w = orig_weight.shape

                        # 选择前out_c个输出通道和前in_c个输入通道
                        new_weight = orig_weight[:out_c, :in_c, :, :]
                        new_state_dict[key] = new_weight
                        show_print(f"卷积层剪枝: {key}, 选择输出通道 [0:{out_c}], 输入通道 [0:{in_c}]")

                elif 'bn' in key and (
                        'weight' in key or 'bias' in key or 'running_mean' in key or 'running_var' in key):
                    # 批归一化层处理
                    if orig_weight.shape[0] > pruned_shape[0]:
                        # 选择前n个通道的权重
                        new_weight = orig_weight[:pruned_shape[0]]
                        new_state_dict[key] = new_weight
                        show_print(f"BN层剪枝: {key}, 选择前 {pruned_shape[0]} 个通道")
                    else:
                        # 使用默认初始化
                        new_state_dict[key] = pruned_state_dict[key]
                        show_print(f"BN层使用默认初始化: {key}")

                elif 'fc' in key and 'weight' in key:
                    # 全连接层权重处理
                    if key == 'fc.weight':
                        # 特殊处理主fc层，连接卷积输出到512维
                        orig_out, orig_in = orig_weight.shape
                        pruned_out, pruned_in = pruned_shape

                        # 选择前pruned_in个输入特征和前pruned_out个输出特征
                        new_weight = orig_weight[:pruned_out, :pruned_in]
                        new_state_dict[key] = new_weight
                        show_print(f"FC层剪枝: {key}, 选择输出 [{pruned_out}] 输入 [{pruned_in}]")
                    else:
                        # 其他全连接层
                        if orig_weight.shape == pruned_shape:
                            new_state_dict[key] = orig_weight.clone()
                        else:
                            new_state_dict[key] = pruned_state_dict[key]

                elif 'fc' in key and 'bias' in key:
                    # 全连接层偏置
                    if key == 'fc.bias':
                        orig_bias = orig_weight
                        pruned_bias_shape = pruned_shape[0]
                        new_bias = orig_bias[:pruned_bias_shape]
                        new_state_dict[key] = new_bias
                        show_print(f"FC偏置剪枝: {key}, 选择前 {pruned_bias_shape} 个元素")
                    else:
                        new_state_dict[key] = pruned_state_dict[key]
                else:
                    # 其他层使用默认初始化
                    new_state_dict[key] = pruned_state_dict[key]
                    show_print(f"其他层使用默认初始化: {key}")
        else:
            # 新层使用默认初始化
            new_state_dict[key] = pruned_state_dict[key]
            show_print(f"新层使用默认初始化: {key}")

    # 加载新的状态字典
    pruned_model.load_state_dict(new_state_dict, strict=False)

    print("权重复制完成")
    return pruned_model

def custom_prune_model(model_path: str, custom_pruning_file, prune_ranks_path, preprocess_type, show_model_summary=False):
    """自定义剪枝模型

根据指定的剪枝率文件对模型进行自定义剪枝，构建剪枝后的新模型并复制权重。

Args:
    model_path (str): 原始模型文件的路径
    show_model_summary: 是否显示模型摘要

Returns:
    PrunedModel: 剪枝后的模型实例

Note:
    - 剪枝率文件应为CSV格式，每行一个剪枝率值
    - 实际应用中需要根据具体模型结构修改PrunedModel类的实现
    - ranks_path参数必须是l2_idx.csv文件，用于自动剪枝流程
"""

    # 加载剪枝率
    r = np.loadtxt(custom_pruning_file, delimiter=",")
    r = [1 - x for x in r]

    # 创建剪枝模型（PrunedRSNet）
    pruned_embedding_net = pruned_drsnet18(r, in_channels=1 if preprocess_type == 0 else 2)

    # 加载原始模型
    original_model = load_model(model_path, NetworkType.WAVELET.value, preprocess_type)

    # 复制权重到剪枝模型
    pruned_embedding_net = copy_weights(original_model.embedding_net, pruned_embedding_net,
                                             prune_ranks_path, show_model_summary)

    # 计算原始模型统计
    print("\n计算原始模型统计...")
    original_stats = compute_model_stats(
        original_model,
        "original",
        input_shape=(1 if preprocess_type == 0 else 2, 256)
    )

    # 包装成 TripletNet
    pruned_triplet_net = TripletNet(
        net_type=2,
        in_channels=1 if preprocess_type == 0 else 2,
        custom_pruning_file=custom_pruning_file,
    )

    # 将剪枝后的权重加载到 TripletNet 的 embedding_net
    pruned_triplet_net.embedding_net.load_state_dict(pruned_embedding_net.state_dict())

    return pruned_triplet_net

def finetune_model_pytorch(model, data_train, labels_train, data_valid, labels_valid,
                          checkpoint_dir: str, exit_epoch: int, batch_size: int, verbose: int):
    """微调PyTorch模型"""

    # 数据加载器
    train_dataset = TripletDataset(data_train, labels_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataset = TripletDataset(data_valid, labels_valid)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)

    # 优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    loss_fn = TripletLoss(margin=0.1)

    model.to(DEVICE)
    best_val_loss = float('inf')

    num_epochs = 150
    batch_num = math.ceil(len(train_dataset) / batch_size)

    patience = 10
    patience_counter = 0


    print(
        "\n---------------------\n"
        "Num of epoch: {}\n"
        "Batch size: {}\n"
        "Num of train batch: {}\n"
        "---------------------\n".format(num_epochs, batch_size, batch_num)
    )
    loss_per_epoch = []

    history = {'loss': [], 'val_loss': [], 'val_accuracy': []}

    # 总进度条
    with tqdm(total=num_epochs, desc=f"Extractor_{exit_epoch}.pth") as total_bar:
        for epoch in range(num_epochs):
            start_time_ep = time.time()
            total_loss = 0.0

            # 训练阶段
            model.to(DEVICE)
            model.train()

            # 每一轮训练进度条
            with tqdm(total=batch_num, desc=f"Epoch {epoch}", leave=False) as pbar:
                for batch_idx, (anchor, positive, negative) in enumerate(train_loader):

                    anchor, positive, negative = (
                        anchor.to(DEVICE),
                        positive.to(DEVICE),
                        negative.to(DEVICE),
                    )

                    # 前向传播
                    embedded_anchor, embedded_positive, embedded_negative = model(
                        anchor, positive, negative
                    )

                    loss = loss_fn(
                        embedded_anchor, embedded_positive, embedded_negative
                    )

                    # 反向传播与优化
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    pbar.update(1)

            # 验证阶段
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for batch_idx, (anchor, positive, negative) in enumerate(valid_loader):
                    anchor = anchor.to(DEVICE)
                    positive = positive.to(DEVICE)
                    negative = negative.to(DEVICE)

                    # 向前传播
                    embedded_anchor, embedded_positive, embedded_negative = model(
                        anchor, positive, negative
                    )

                    loss = loss_fn(
                        embedded_anchor, embedded_positive, embedded_negative
                    )

                    val_loss += loss.item()

                    # 准确率计算：使用嵌入向量的相似度进行验证
                    # 计算anchor和positive之间的相似度
                    pos_similarity = F.cosine_similarity(embedded_anchor, embedded_positive)
                    neg_similarity = F.cosine_similarity(embedded_anchor, embedded_negative)

                    # 如果positive比negative更相似，则认为预测正确
                    predictions = (pos_similarity > neg_similarity).float()
                    batch_correct = predictions.sum().item()

                    correct += batch_correct
                    total += anchor.size(0)

                # 计算平均损失
                avg_train_loss = total_loss / len(train_loader)
                avg_val_loss = val_loss / len(valid_loader)
                val_accuracy = 100 * correct / total if total > 0 else 0.0

                # 记录历史
                history['loss'].append(avg_train_loss)
                history['val_loss'].append(avg_val_loss)
                history['val_accuracy'].append(val_accuracy)

                text = (
                    f'Train Loss: {avg_train_loss:.4f}, '+
                    f'Val Loss: {avg_val_loss:.4f}, '+
                    f'Val Acc: {val_accuracy:.2f}%'
                )
                tqdm.write(text)

                if verbose:
                    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, '
                          f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')

                # 早停和模型保存
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    torch.save(model.state_dict(), checkpoint_dir)
                    patience_counter = 0
                    if verbose:
                        print(f"保存最佳模型，验证损失: {best_val_loss:.4f}")
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    if verbose:
                        print("早停触发")
                    break

                # 记录每轮损失
                loss_per_epoch.append(avg_train_loss)
            # 更新进度条
            total_bar.update(1)

    # 加载最佳模型
    if os.path.exists(checkpoint_dir):
        model.load_state_dict(torch.load(checkpoint_dir))
        if verbose:
            print(f"加载最佳模型: {checkpoint_dir}")

    return model, history

def gratitude_pr(rank_result, n, out_dir: str, epoch):
    """基于梯度的剪枝"""
    gratitude = []
    pruning_rate = []
    idxs = []

    for rank in rank_result:
        rank = rank[1:]
        gra = []
        for idx, r in enumerate(rank):
            if idx == len(rank) - n:
                break
            try:
                g = (rank[idx + n] - r) / n  # 计算梯度
                gra.append(g)
            except IndexError:
                pass

        gratitude.append(np.array(gra))

    for gra in gratitude:
        for idx, g in enumerate(gra):
            if g == max(gra):
                idxs.append(int(idx + n / 2))  # 找到最大梯度的索引
                pruning_rate.append(
                    float("{:.2f}".format(1 - (int(idx + n / 2)) / len(gra)))
                )  # 计算剪枝率
                break

    for i in range(len(pruning_rate)):
        if pruning_rate[i] > 0.9:
            pruning_rate[i] = 0.9
        elif pruning_rate[i] < 0:
            pruning_rate[i] = 0

    # 转换为numpy数组并保存
    pruning_rate = np.array(pruning_rate)
    np.savetxt(os.path.join(out_dir, f"Extractor_{epoch}_1-pr.csv"), pruning_rate, delimiter=",")
    print(f"  - 保存文件: {os.path.join(out_dir, f"Extractor_{epoch}_1-pr.csv")}")
    return pruning_rate

def automatic_pruner_pytorch(n: int, output: str, epoch):
    """调用梯度剪枝"""
    rank_result = pd.read_csv(output, header=None).values
    rr = []
    for r in rank_result:
        r = r[~np.isnan(r)]
        rr.append(r)

    # 调用gratitude_pr函数
    pruning_rates = gratitude_pr(rr, n, PRUNED_OUTPUT_DIR, epoch)
    return pruning_rates
