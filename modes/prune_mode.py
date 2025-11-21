"""剪枝模式相关函数"""
import math
import os
import time
from datetime import datetime

import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.config import Config, PRUNED_OUTPUT_DIR, H_VAL, DEVICE
from experiment_logger import ExperimentLogger
from modes.classification_mode import test_classification
from plot.loss_plot import plot_loss_curve
from pruning_utils.Pytorch_prunner import pytorch_native_prune, remove_pruning_masks_and_apply, \
    compute_pruning_rates, pytorch_prune_model
from pruning_utils.TinyML_prunner import extract_weight, automatic_pruner_pytorch, TinyML_prune_model
from training_utils.TripletDataset import TripletDataset, TripletLoss
from training_utils.data_preprocessor import load_model
from utils.better_print import print_colored_text
from utils.yaml_handler import update_nested_yaml_entry


def pruning(
        data,
        labels,
        config: Config,
        model_dir: str,
        net_type=None,
        preprocess_type=None,
        test_list: list=None,
        batch_size=32,
        show_model_summary=False,
        skip_finetune=False,
        verbose=True,
        training_verbose=True,
        use_pytorch_prune=True,  # 新增：是否使用PyTorch原生剪枝
        target_sparsity=0.5,  # 新增：目标稀疏度
):
    """

    :param data: 训练数据。
    :param labels: 训练标签
    :param config: 配置文件
    :param model_dir: 模型目录路径
    :param net_type: 网络类型
    :param preprocess_type: 预处理类型
    :param test_list: 测试点列表
    :param batch_size: 减小批大小以适应小数据集
    :param show_model_summary: 是否显示模型摘要
    :param skip_finetune: 是否跳过微调
    :param verbose: 是否显示详细信息
    :param training_verbose: 训练时是否显示详细信息
    :param use_pytorch_prune: 是否使用PyTorch原生剪枝
    :param target_sparsity: 目标稀疏度 (0-1)
    """

    # 初始化实验记录
    logger = ExperimentLogger()
    exp_config = {
        "mode": "pruning",
        "model": {
            "type": net_type,
            "parameters": {
                "batch_size": batch_size,
                "target_sparsity": target_sparsity,
                "use_pytorch_prune": use_pytorch_prune
            }
        },
        "data": {
            "preprocess_type": preprocess_type,
            "test_points": test_list
        }
    }
    exp_filepath, exp_id = logger.create_experiment_record(exp_config)

    # 定义YAML文件路径
    yaml_file_path = os.path.join(model_dir, "performance_records.yaml")

    # 执行剪枝
    print("开始模型剪枝...")
    if use_pytorch_prune:
        print(">>> 使用 PyTorch 原生剪枝方案 <<<")
    else:
        print(">>> 使用特定的 TinyML 剪枝方案 <<<")

    pruning_results = {}

    for exit_epoch in test_list or []:
        print(exit_epoch, test_list)
        print()
        print("=============================")
        origin_model_dir = os.path.join(model_dir, f"origin/Extractor_{exit_epoch}.pth")
        pruned_model_dir = os.path.join(model_dir, f"prune/Extractor_{exit_epoch}.pth")

        if not os.path.exists(origin_model_dir):
            print(f"{origin_model_dir} isn't exist")
        else:
            # 加载模型和数据
            print("\n加载模型...")

            original_model = load_model(origin_model_dir, config.NET_TYPE, preprocess_type)
            if show_model_summary:
                print("模型结构:")
                print(original_model)

            print("\n加载数据...")

            # 数据集划分
            data_train, data_valid, labels_train, labels_valid = train_test_split(
                data, labels, test_size=0.1, shuffle=True, random_state=42
            )

            if verbose:
                tot = data_train.shape[0] + data_valid.shape[0]
                print(f"  - 总IQ轨迹数: {tot}")
                print(f"  - 训练数据形状: {data_train.shape}")
                print(f"  - 验证数据形状: {data_valid.shape}")

            prune_start = datetime.now()

            if use_pytorch_prune:
                # ========== PyTorch原生剪枝路径 ==========

                # 计算剪枝率
                print("\n[Pytorch剪枝] 计算剪枝率...")
                pruning_rates = compute_pruning_rates(
                    original_model.embedding_net,
                    target_sparsity,
                    method='tinyml_gradient',
                    show_model_summary=show_model_summary
                )

                # 应用剪枝
                print("\n[Pytorch剪枝] 应用剪枝...")
                pruned_embedding_net = pytorch_native_prune(
                    original_model.embedding_net,
                    pruning_rates,
                    method='ln',
                    show_model_summary=show_model_summary
                )

                # 使剪枝永久化并获取清理后的状态字典
                print("\n[Pytorch剪枝] 使剪枝永久化...")
                pruned_embedding_net = remove_pruning_masks_and_apply(pruned_embedding_net)
                # 这里的 pruned_embedding_net 只是经过剪枝处理的 embedding_net 部分, 缺少完整的 TripletNet 结构包装

                print("\n[Pytorch剪枝] 创建剪枝模型...")
                pruned_model = pytorch_prune_model(pruned_embedding_net, net_type, preprocess_type)  # 确保结构一致

                # 剪枝时间
                prune_runtime = datetime.now() - prune_start

                if verbose:
                    print(f"剪枝运行时间: {prune_runtime.total_seconds():.2f}秒")

                # 收集剪枝阶段的统计信息
                pruning_info = {
                    'epoch': exit_epoch,
                    'prune_runtime': prune_runtime.total_seconds(),
                    'use_pytorch_prune': use_pytorch_prune,
                    'target_sparsity': target_sparsity,
                    'timestamp': datetime.now().isoformat()
                }

                pruning_results[f"epoch_{exit_epoch}"] = {
                    "pruning_info": pruning_info
                }

                # 剪枝信息写入：
                update_nested_yaml_entry(
                    yaml_file_path,
                    [f'models', 'pruned', 'pruning_history', f'epoch{exit_epoch}'],
                    pruning_info
                )

            else:
                # ========== TinyML剪枝路径 ==========
                prune_ranks_path = PRUNED_OUTPUT_DIR + f"Extractor_{exit_epoch}_l2_idx.csv"
                prune_rank_path = PRUNED_OUTPUT_DIR + f"Extractor_{exit_epoch}_l2.csv"
                custom_pruning_file = os.path.join(PRUNED_OUTPUT_DIR, f"Extractor_{exit_epoch}_1-pr.csv")

                # 生成剪枝排名
                print("\n[TinyML剪枝] 生成剪枝排名...")
                extract_weight(original_model, prune_rank_path, prune_ranks_path, show_model_summary)

                # 生成剪枝文件
                print("\n[TinyML剪枝] 生成剪枝文件...")
                automatic_pruner_pytorch(H_VAL, prune_rank_path, exit_epoch)

                # 应用TinyML剪枝率创建剪枝模型
                print("\n[TinyML剪枝] 创建剪枝模型...")
                pruned_model = TinyML_prune_model(
                    original_model, custom_pruning_file, prune_ranks_path, preprocess_type, show_model_summary
                )

            if show_model_summary:
                print("剪枝后模型结构:")
                print(pruned_model)

            if not skip_finetune:
                print("\n开始微调...")
                finetune_start = datetime.now()

                # 微调剪枝模型
                finetuned_pruned_model, history = finetune_model(
                    pruned_model, data_train, labels_train, data_valid, labels_valid,
                    net_type, preprocess_type,
                    checkpoint_dir=pruned_model_dir, exit_epoch=exit_epoch,
                    batch_size=batch_size, verbose=1 if training_verbose else 0
                )
                # 微调时间
                finetune_runtime = datetime.now() - finetune_start

                print(f"微调时间: {finetune_runtime.total_seconds():.2f}秒")
                print(f"微调轮数: {len(history['loss'])}")
                print(f"每轮平均时间: {finetune_runtime.total_seconds() / len(history['loss']):.2f}秒")

                # 微调完成后，更新统计信息
                finetune_info = {
                    'epoch': exit_epoch,
                    'finetune_runtime': finetune_runtime.total_seconds(),
                    'final_val_loss': history['best_val_loss'],
                    'num_epochs_trained': len(history['loss']) if history and 'loss' in history else 0
                }

                if f"epoch_{exit_epoch}" in pruning_results:
                    pruning_results[f"epoch_{exit_epoch}"]["finetune_info"] = finetune_info
                else:
                    pruning_results[f"epoch_{exit_epoch}"] = {"finetune_info": finetune_info}

                # 微调信息写入：
                update_nested_yaml_entry(
                    yaml_file_path,
                    [f'models', 'pruned', 'finetune_history', f'epoch{exit_epoch}'],
                    finetune_info
                )

            else:
                # 即使跳过微调也需要保存剪枝后的模型
                torch.save(pruned_model.state_dict(), pruned_model_dir)
                print("跳过微调步骤，已保存剪枝模型")

    # 记录实验结果
    final_results = {
        "pruning": pruning_results,
        "model_saved_path": model_dir
    }
    logger.update_experiment_result(exp_id, final_results)

    return

def finetune_model(
                model,
                data_train,
                labels_train,
                data_valid,
                labels_valid,
                net_type,
                preprocess_type,
                checkpoint_dir: str,
                exit_epoch: int,
                batch_size: int,
                verbose: int
):
    """
    微调剪枝后的模型

    :param model: 待微调的模型
    :param data_train: 训练数据
    :param labels_train: 训练标签
    :param data_valid: 验证数据
    :param labels_valid: 验证标签
    :param net_type: 网络类型
    :param preprocess_type: 预处理类型
    :param checkpoint_dir: 模型检查点保存路径
    :param exit_epoch: 退出轮次
    :param batch_size: 批处理大小
    :param verbose: 详细输出级别 (0: 静默, 1: 基本信息, 2: 详细信息)
    """

    # 参数配置
    patience = 10
    patience_counter = 0
    model.to(DEVICE)
    best_val_loss = float('inf')
    num_epochs = 150
    learning_rate=1e-4

    # 数据加载器
    train_dataset = TripletDataset(data_train, labels_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataset = TripletDataset(data_valid, labels_valid)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)

    # 优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = TripletLoss(margin=0.1)

    batch_num = math.ceil(len(train_dataset) / batch_size)


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

                # 早停和模型保存

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    # 保存模型到指定路径
                    torch.save(model.state_dict(), checkpoint_dir)
                    patience_counter = 0
                    print(f"保存最佳模型，验证损失: {best_val_loss:.4f}")
                    tqdm.write(f"Model saved to {checkpoint_dir}")

                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    if verbose:
                        print("-- 早停触发!!! --")
                    history['best_val_loss'] = best_val_loss
                    break

                # 记录每轮损失
                loss_per_epoch.append(avg_train_loss)

                # 绘制loss折线图
                pic_save_path = checkpoint_dir + f"loss_{epoch+1}.png"
                plot_loss_curve(loss_per_epoch, num_epochs, net_type, preprocess_type, pic_save_path)
            # 更新进度条
            total_bar.update(1)

    # 加载最佳模型
    if os.path.exists(checkpoint_dir):
        model.load_state_dict(torch.load(checkpoint_dir))
        if verbose:
            print(f"加载最佳模型: {checkpoint_dir}")

    return model, history