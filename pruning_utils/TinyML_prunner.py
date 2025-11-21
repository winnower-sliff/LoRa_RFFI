"""
Author: Ryan
结合 rank_gen.py 和 automatic_prunner.py
用于计时整个剪枝过程，而不是rank_gen时间 + automatic_pruner时间

"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# 从配置模块导入
from core.config import NetworkType, PRUNED_OUTPUT_DIR
from net.TripletNet import TripletNet
from net.net_prune import pruned_drsnet18
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

def TinyML_prune_model(original_model: nn.Module, custom_pruning_file, prune_ranks_path, preprocess_type, show_model_summary=False):
    """TinyML_prune_model

    根据指定的剪枝率文件对模型进行自定义剪枝，构建剪枝后的新模型并复制权重。

    Args:
        original_model: 原始模型
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

    # 复制权重到剪枝模型
    pruned_embedding_net = copy_weights(original_model.embedding_net, pruned_embedding_net,
                                             prune_ranks_path, show_model_summary)

    # 包装成 TripletNet
    pruned_triplet_net = TripletNet(
        net_type=2,
        in_channels=1 if preprocess_type == 0 else 2,
        custom_pruning_file=custom_pruning_file,
    )

    # 将剪枝后的权重加载到 TripletNet 的 embedding_net
    pruned_triplet_net.embedding_net.load_state_dict(pruned_embedding_net.state_dict())

    # 计算剪枝模型统计
    try:
        print("\n计算剪枝模型统计...")
        original_stats = compute_model_stats(
            pruned_triplet_net,
            "original",
            input_shape=(1 if preprocess_type == 0 else 2, 256, 256)
        )

        # 添加输出原始模型统计信息
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
