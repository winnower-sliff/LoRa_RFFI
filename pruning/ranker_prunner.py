"""
Author: Ryan
结合 rank_gen.py 和 automatic_prunner.py
用于计时整个剪枝过程，而不是rank_gen时间 + automatic_pruner时间

"""
import math
import os
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.config import Config, DEVICE, PruneType, NetworkType
from net.TripletNet import TripletNet
from net.net_prune import pruned_drsnet18
from training_utils.TripletDataset import TripletDataset, TripletLoss
from training_utils.data_preprocessor import load_model


class PyTorchPruner:
    """PyTorch剪枝器主类"""

    def __init__(self, config: Config = None, output_dir:str = None):
        '''

        Args:
            config:
            output_dir:
        content:
            custom_pruning_file (str): 剪枝率配置文件路径，包含各层的剪枝比例
            ranks_path (str): 排名文件路径，应为l2_idx.csv格式，用于权重复制
        '''
        self.config = config
        self.device = DEVICE
        self.output_dir = output_dir


    def plot_confusion_matrix(self, cm, path, title='混淆矩阵', cmap=plt.cm.Blues, labels=[]):
        """绘制混淆矩阵"""
        plt.figure(figsize=(15, 10))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        label_len = np.shape(labels)[0]
        tick_marks = np.arange(label_len)
        plt.xticks(tick_marks, labels, rotation=45)
        plt.yticks(tick_marks, labels)
        plt.tight_layout()
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
        plt.savefig(path)
        plt.close()

    def mapping(self, W, min_w, max_w):
        """权重映射函数"""
        scale_w = (max_w - min_w) / 100
        min_arr = torch.full(W.shape, min_w, device=W.device)
        q_w = torch.round((W - min_arr) / scale_w).to(torch.uint8)
        return q_w

    def extract_weight(self, model: nn.Module, prune_rank_path: str, prune_ranks_path: str):
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
                r = self.mapping(r, torch.min(r), torch.max(r))
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

    def fpgm(self, model: nn.Module, output_dir: Path, dist_type="l2"):
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
                r = self.mapping(torch.tensor(distance_sum),
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

    def copy_weights(self, original_model: nn.Module, pruned_model: nn.Module, ranks_path: str):
        """复制权重到剪枝后的模型"""

        # 获取状态字典
        original_state_dict = original_model.state_dict()
        pruned_state_dict = pruned_model.state_dict()

        # 打印模型结构信息用于调试
        print("\n原始模型结构:")
        for name, param in original_model.named_parameters():
            print(f"  {name}: {param.shape}")

        print("\n剪枝模型结构:")
        for name, param in pruned_model.named_parameters():
            print(f"  {name}: {param.shape}")

        # 权重复制逻辑
        new_state_dict = {}

        for key in pruned_state_dict.keys():
            if key in original_state_dict:
                orig_weight = original_state_dict[key]
                pruned_shape = pruned_state_dict[key].shape

                if orig_weight.shape == pruned_shape:
                    # 形状相同，直接复制
                    new_state_dict[key] = orig_weight.clone()
                    print(f"直接复制: {key}")
                else:
                    # 形状不同，需要选择性复制
                    print(f"形状不匹配 - {key}: {orig_weight.shape} -> {pruned_shape}")

                    # 根据不同的层类型处理权重复制
                    if 'conv' in key and 'weight' in key:
                        # 卷积层权重处理
                        if len(orig_weight.shape) == 4:  # Conv2d权重
                            out_c, in_c, h, w = pruned_shape
                            orig_out_c, orig_in_c, orig_h, orig_w = orig_weight.shape

                            # 选择前out_c个输出通道和前in_c个输入通道
                            new_weight = orig_weight[:out_c, :in_c, :, :]
                            new_state_dict[key] = new_weight
                            print(f"卷积层剪枝: {key}, 选择输出通道 [0:{out_c}], 输入通道 [0:{in_c}]")

                    elif 'bn' in key and (
                            'weight' in key or 'bias' in key or 'running_mean' in key or 'running_var' in key):
                        # 批归一化层处理
                        if orig_weight.shape[0] > pruned_shape[0]:
                            # 选择前n个通道的权重
                            new_weight = orig_weight[:pruned_shape[0]]
                            new_state_dict[key] = new_weight
                            print(f"BN层剪枝: {key}, 选择前 {pruned_shape[0]} 个通道")
                        else:
                            # 使用默认初始化
                            new_state_dict[key] = pruned_state_dict[key]
                            print(f"BN层使用默认初始化: {key}")

                    elif 'fc' in key and 'weight' in key:
                        # 全连接层权重处理
                        if key == 'fc.weight':
                            # 特殊处理主fc层，连接卷积输出到512维
                            orig_out, orig_in = orig_weight.shape
                            pruned_out, pruned_in = pruned_shape

                            # 选择前pruned_in个输入特征和前pruned_out个输出特征
                            new_weight = orig_weight[:pruned_out, :pruned_in]
                            new_state_dict[key] = new_weight
                            print(f"FC层剪枝: {key}, 选择输出 [{pruned_out}] 输入 [{pruned_in}]")
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
                            print(f"FC偏置剪枝: {key}, 选择前 {pruned_bias_shape} 个元素")
                        else:
                            new_state_dict[key] = pruned_state_dict[key]
                    else:
                        # 其他层使用默认初始化
                        new_state_dict[key] = pruned_state_dict[key]
                        print(f"其他层使用默认初始化: {key}")
            else:
                # 新层使用默认初始化
                new_state_dict[key] = pruned_state_dict[key]
                print(f"新层使用默认初始化: {key}")

        # 加载新的状态字典
        pruned_model.load_state_dict(new_state_dict, strict=False)

        print("权重复制完成")
        return pruned_model

    def custom_prune_model(self, model_path: str, prune_ranks_path, num_classes=10):
        """自定义剪枝模型

    根据指定的剪枝率文件对模型进行自定义剪枝，构建剪枝后的新模型并复制权重。

    Args:
        model_path (str): 原始模型文件的路径
        num_classes (int, optional): 分类任务的类别数，默认为10

    Returns:
        PrunedModel: 剪枝后的模型实例

    Note:
        - 剪枝率文件应为CSV格式，每行一个剪枝率值
        - 实际应用中需要根据具体模型结构修改PrunedModel类的实现
        - ranks_path参数必须是l2_idx.csv文件，用于自动剪枝流程
    """

        # 加载剪枝率
        r = np.loadtxt(self.config.custom_pruning_file, delimiter=",")
        r = [1 - x for x in r]

        # 创建剪枝模型（PrunedRSNet）
        pruned_embedding_net = pruned_drsnet18(r, in_channels=1 if self.config.PROPRECESS_TYPE == 0 else 2)

        # 加载原始模型
        original_model = load_model(model_path, NetworkType.WAVELET.value, self.config.PROPRECESS_TYPE)

        # 复制权重到剪枝模型
        pruned_embedding_net = self.copy_weights(original_model.embedding_net, pruned_embedding_net,
                                                 prune_ranks_path)

        # 包装成 TripletNet
        pruned_triplet_net = TripletNet(
            net_type=2,
            in_channels=1 if self.config.PROPRECESS_TYPE == 0 else 2,
            config=self.config
        )

        # 将剪枝后的权重加载到 TripletNet 的 embedding_net
        pruned_triplet_net.embedding_net.load_state_dict(pruned_embedding_net.state_dict())

        return pruned_triplet_net

    def finetune_model_pytorch(self, model, data_train, labels_train, data_valid, labels_valid,
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

        model.to(self.device)
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
                        anchor = anchor.to(self.device)
                        positive = positive.to(self.device)
                        negative = negative.to(self.device)

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

    def gratitude_pr(self, rank_result, n, out_dir: str, epoch):
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
        print(f"保存文件: 1-pr.csv")
        return pruning_rate

    def automatic_pruner_pytorch(self, n: int, output: str, epoch):
        """调用梯度剪枝"""
        rank_result = pd.read_csv(output, header=None).values
        rr = []
        for r in rank_result:
            r = r[~np.isnan(r)]
            rr.append(r)

        # 调用gratitude_pr函数
        pruning_rates = self.gratitude_pr(rr, n, self.output_dir, epoch)
        return pruning_rates

    def prune_automatic(self, data, labels,
                        origin_model_dir: str,
                        pruned_model_dir: str,
                        test_list: list,
                        prune_type: PruneType = PruneType.l2,
                        batch_size=32,
                        h_val: int = 10,
                        show_model_summary: bool = True,
                        skip_finetune: bool = False,
                        verbose: bool = True,
                        training_verbose: bool = True
                    ):
        """
        自动剪枝主函数

        参数:
            origin_model_dir: 原始模型路径
            pruned_model_dir: 剪枝后模型路径
            prune_type: 剪枝类型
            batch_size: 减小批大小以适应小数据集
            h_val: 剪枝参数
            show_model_summary: 是否显示模型摘要
            skip_finetune: 是否跳过微调
            verbose: 是否显示详细信息
            training_verbose: 训练时是否显示详细信息
        """

        for exit_epoch in test_list or []:
            print(exit_epoch, test_list)
            print()
            print("=============================")
            origin_model_path = origin_model_dir + f"Extractor_{exit_epoch}.pth"
            pruned_model_path = pruned_model_dir + f"Extractor_{exit_epoch}.pth"
            if not os.path.exists(origin_model_path):
                print(f"{origin_model_path} isn't exist")
            else:
                # 加载模型和数据
                print("加载模型...")

                model = load_model(origin_model_path, NetworkType.WAVELET.value, self.config.PROPRECESS_TYPE, config=self.config)
                if show_model_summary:
                    print("模型结构:")
                    print(model)

                print("加载数据...")

                # 数据集划分 - 分为训练集、验证集、测试集
                # 终比例：训练集70 %，验证集20 %，测试集10 %
                data_train, data_temp, labels_train, labels_temp = train_test_split(
                    data, labels, test_size=0.3, shuffle=True, random_state=42  # 30%作为临时数据
                )
                data_valid, data_test, labels_valid, labels_test = train_test_split(
                    data_temp, labels_temp, test_size=0.333, shuffle=True, random_state=42  # 临时数据的1/3作为测试集
                )

                if verbose:
                    tot = data_train.shape[0] + data_valid.shape[0] + data_test.shape[0]
                    print(f"总IQ轨迹数: {tot}")
                    print(f"训练数据形状: {data_train.shape}")
                    print(f"验证数据形状: {data_valid.shape}")
                    print(f"测试数据形状: {data_test.shape}")

                prune_ranks_path = self.output_dir + f"Extractor_{exit_epoch}_l2_idx.csv"
                prune_rank_path = self.output_dir + f"Extractor_{exit_epoch}_l2.csv"
                self.config.custom_pruning_file = os.path.join(self.config.pruned_output_dir, f"Extractor_{exit_epoch}_1-pr.csv")

                # 生成剪枝排名
                print("生成剪枝排名...")
                prune_start = datetime.now()
                self.extract_weight(model, prune_rank_path, prune_ranks_path)

                # 生成剪枝文件
                print("生成剪枝文件...")
                self.automatic_pruner_pytorch(h_val, prune_rank_path, exit_epoch)

                # 应用自定义剪枝率创建剪枝模型
                print("创建剪枝模型...")

                pruned_model = self.custom_prune_model(
                    origin_model_path, prune_ranks_path
                )

                prune_runtime = datetime.now() - prune_start

                if verbose:
                    print(f"剪枝运行时间: {prune_runtime.total_seconds()}秒")

                if show_model_summary:
                    print("剪枝后模型结构:")
                    print(pruned_model)

                if not skip_finetune:
                    print("开始微调...")
                    finetune_start = datetime.now()

                    # 5. 微调剪枝模型
                    finetuned_pruned_model, history = self.finetune_model_pytorch(
                        pruned_model, data_train, labels_train, data_valid, labels_valid,
                        checkpoint_dir=pruned_model_path, exit_epoch=exit_epoch,
                        batch_size=batch_size, verbose=1 if training_verbose else 0
                    )

                    finetune_runtime = datetime.now() - finetune_start

                    print(f"微调时间: {finetune_runtime.total_seconds()}秒")
                    if history and 'loss' in history:
                        print(f"微调轮数: {len(history['loss'])}")
                        if len(history['loss']) > 0:
                            print(f"每轮平均时间: {finetune_runtime.total_seconds()/len(history['loss'])}秒")

                    # print(f"最终准确率: {accuracy:.2f}%")
                else:
                    print("跳过微调步骤")

                print(f"总剪枝运行时间: {prune_runtime.total_seconds()}秒")
        return
