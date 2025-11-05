"""
Author: Ryan
结合 rank_gen.py 和 automatic_pruner.py
用于计时整个剪枝过程，而不是rank_gen时间 + automatic_pruner时间

PyTorch版本 - 与现有框架集成
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import h5py
import os
import json
import pandas as pd
from pathlib import Path
from enum import Enum
from typing import Dict, List, Tuple, Optional, Union
import warnings
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# 导入你的现有模块
try:
    from core.config import Config, DEVICE
    from training_utils.data_preprocessor import prepare_train_data
except ImportError:
    # 如果导入失败，创建模拟配置
    class Config:
        def __init__(self):
            self.MODEL_DIR_PATH = "models"
            self.NET_TYPE = "resnet"
            self.PROPRECESS_TYPE = "standard"
            self.TEST_LIST = [100]
            self.WST_J = 3
            self.WST_Q = 8
            self.PPS_FOR = "classification"

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PruneType(str, Enum):
    l2 = "l2"
    fpgm = "fpgm"


class TestModel(nn.Module):
    """测试用的简单模型"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 16, (3, 1), padding='same')
        self.conv2 = nn.Conv2d(16, 32, (3, 1), padding='same')
        self.conv3 = nn.Conv2d(32, 64, (3, 1), padding='same')
        self.fc = nn.Linear(64 * 1024, 10)  # 假设输入尺寸
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class PyTorchPruner:
    """PyTorch剪枝器主类"""

    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.device = DEVICE

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

    def extract_weight(self, model: nn.Module, output: Path):
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
        output.mkdir(exist_ok=True)
        df = pd.DataFrame(results, index=None)
        df.to_csv(output.joinpath("l2.csv"), header=False, index=False)
        df = pd.DataFrame(idx_results, index=None)
        df.to_csv(output.joinpath("l2_idx.csv"), header=False, index=False)

        print(f"权重提取完成! 生成文件:")
        print(f"  - {output.joinpath('l2.csv')}")
        print(f"  - {output.joinpath('l2_idx.csv')}")

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

    def copy_weights(self, pre_trained_model: nn.Module, target_model: nn.Module, ranks_path: Path):
        """复制权重到目标模型"""
        ranks = pd.read_csv(ranks_path, header=None).values

        rr = []
        for r in ranks:
            r = r[~np.isnan(r)]
            r = list(map(int, r))
            rr.append(r)

        i = 0
        last_filters = None

        # 获取模型的所有层
        pre_layers = dict(pre_trained_model.named_modules())
        target_layers = dict(target_model.named_modules())

        for target_name, target_layer in target_layers.items():
            if isinstance(target_layer, (nn.Conv2d, nn.Linear)):
                if i == 0 and isinstance(target_layer, nn.Conv2d):
                    i += 1
                    continue  # 跳过第一个Conv2D层

                conv_id = i - 1 if isinstance(target_layer, nn.Conv2d) else None
                if conv_id is not None and conv_id >= len(rr):
                    print(f"错误: conv_id {conv_id} 超出范围")
                    break

                if conv_id is not None:
                    this_indices = rr[conv_id][:target_layer.out_channels]
                    this_indices = np.clip(this_indices, 0, target_layer.out_channels - 1)
                    print(f"卷积层 {i}: {target_name}, 索引: {this_indices}")
                else:
                    this_indices = None

                try:
                    if isinstance(target_layer, nn.Conv2d) and target_name in pre_layers:
                        pre_layer = pre_layers[target_name]
                        pre_weights = pre_layer.weight.data
                        pre_bias = pre_layer.bias.data if pre_layer.bias is not None else None

                        if conv_id == 0:
                            # 第一层卷积
                            weights = pre_weights[this_indices, :, :, :]
                            if pre_bias is not None:
                                bias = pre_bias[this_indices]
                        else:
                            # 后续卷积层
                            last_indices = rr[conv_id - 1][:last_filters]
                            last_indices = np.clip(last_indices, 0, last_filters - 1)
                            weights = pre_weights[this_indices, :, :, :][:, last_indices, :, :]

                            # 填充处理
                            pad_width = target_layer.out_channels - len(this_indices)
                            if pad_width > 0:
                                padding = torch.zeros(pad_width, weights.size(1),
                                                    weights.size(2), weights.size(3),
                                                    device=weights.device)
                                weights = torch.cat([weights, padding], dim=0)

                            if pre_bias is not None:
                                bias = pre_bias[this_indices]
                                if pad_width > 0:
                                    bias_padding = torch.zeros(pad_width, device=bias.device)
                                    bias = torch.cat([bias, bias_padding], dim=0)

                        target_layer.weight.data = weights
                        if pre_bias is not None and target_layer.bias is not None:
                            target_layer.bias.data = bias

                        last_filters = target_layer.out_channels
                        i += 1

                    elif isinstance(target_layer, nn.Linear) and target_name in pre_layers:
                        # 全连接层直接复制
                        pre_layer = pre_layers[target_name]
                        target_layer.weight.data = pre_layer.weight.data
                        if pre_layer.bias is not None and target_layer.bias is not None:
                            target_layer.bias.data = pre_layer.bias.data

                except Exception as e:
                    print(f"设置层 {target_name} 权重时出错: {e}")
                    continue

        return target_model

    def load_radio_mod_data(self, dataset: Path):
        """加载无线电调制数据"""
        file_handle = h5py.File(dataset, "r")

        new_myData = file_handle["X"][:]  # 1024x2 samples
        new_myMods = file_handle["Y"][:]  # mods
        new_mySNRs = file_handle["Z"][:]  # snrs

        file_handle.close()

        myData = []
        myMods = []
        mySNRs = []

        # 定义阈值
        threshold = 6
        for i in range(len(new_mySNRs)):
            if new_mySNRs[i] >= threshold:
                myData.append(new_myData[i])
                myMods.append(new_myMods[i])
                mySNRs.append(new_mySNRs[i])

        # 转换为NumPy数组
        myData = np.array(myData)
        myMods = np.array(myMods)
        mySNRs = np.array(mySNRs)

        print(f"数据形状: {np.shape(myData)}")
        print(f"调制形状: {np.shape(myMods)}")
        print(f"SNR形状: {np.shape(mySNRs)}")

        myData = myData.reshape(myData.shape[0], 2, 1024)  # PyTorch格式: (batch, channels, length)

        # 数据分割
        X_train, X_temp, Y_train, Y_temp = train_test_split(
            myData, myMods, test_size=0.2, random_state=0
        )

        X_val, X_test, Y_val, Y_test = train_test_split(
            X_temp, Y_temp, test_size=0.5, random_state=0
        )

        del myData, myMods, mySNRs

        # 转换为PyTorch张量
        X_train = torch.FloatTensor(X_train)
        Y_train = torch.LongTensor(Y_train)
        X_val = torch.FloatTensor(X_val)
        Y_val = torch.LongTensor(Y_val)
        X_test = torch.FloatTensor(X_test)
        Y_test = torch.LongTensor(Y_test)

        return X_train, Y_train, X_val, Y_val, X_test, Y_test

    def custom_prune_model(self, model_path: Path, custom_pruning_file: Path,
                                 ranks_path: Path, X_train, Y_train, num_classes=10):
        """PyTorch版本的自定义剪枝模型"""

        if "_idx.csv" not in ranks_path.name:
            warnings.warn(" ranks路径应该是l2_idx.csv文件")
            warnings.warn(" 在bash历史文件中... l2_idx.csv作为-rp参数传递")
            warnings.warn(" 而... l2.csv作为-i参数传递")
            warnings.warn(" 在代码中两者都被称为ranks_path但使用方式不同:(")
            warnings.warn(" 参数-rp (l2_idx.csv) 用于自动训练，而-i (l2.csv) 用于自动剪枝")
            raise Exception(" ranks路径必须是l2_idx.csv文件!...我认为")

        # 加载剪枝率
        r = np.loadtxt(custom_pruning_file, delimiter=",")
        r = [1 - x for x in r]

        # 这里需要根据你的实际模型结构来构建剪枝后的模型
        # 由于我不知道你的具体模型结构，这里提供一个通用模板
        class PrunedModel(nn.Module):
            def __init__(self, r, num_classes=10):
                super().__init__()
                # 根据剪枝率r构建模型
                # 这里需要根据你的实际模型结构进行修改
                self.conv1 = nn.Conv2d(2, int(16 * r[0]), (5, 1), padding='same')
                self.bn1 = nn.BatchNorm2d(int(16 * r[0]))
                self.relu = nn.ReLU()

                # 添加更多层...
                self.flatten = nn.Flatten()
                self.dropout = nn.Dropout(0.5)
                self.fc = nn.Linear(int(16 * r[0]) * 512, num_classes)  # 假设输入尺寸
                self.softmax = nn.Softmax(dim=1)

            def forward(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu(x)
                # 更多层的前向传播...
                x = self.flatten(x)
                x = self.dropout(x)
                x = self.fc(x)
                x = self.softmax(x)
                return x

        # 创建剪枝模型
        model_pruned = PrunedModel(r, num_classes)

        # 加载原始模型
        original_model = self.load_model(model_path)

        # 复制权重
        model_pruned = self.copy_weights(original_model, model_pruned, ranks_path)

        return model_pruned

    def load_model(self, model_path: Path) -> nn.Module:
        """加载模型，支持state_dict和完整模型两种格式"""
        try:
            # 首先尝试加载完整模型
            model = torch.load(model_path, map_location=self.device)
            if isinstance(model, nn.Module):
                return model
            else:
                # 如果是state_dict，创建模型并加载权重
                model = TestModel()
                model.load_state_dict(model)
                return model.to(self.device)
        except Exception as e:
            print(f"加载模型失败: {e}")
            # 创建新模型
            model = TestModel()
            return model.to(self.device)

    def finetune_model_pytorch(self, model, x_train, y_train, x_val, y_val,
                              checkpoint_dir: Path, batch_size: int, verbose: int):
        """微调PyTorch模型"""

        best_checkpoint = checkpoint_dir.joinpath("pruned_best_checkpoint.pth")

        # 数据加载器
        train_dataset = TensorDataset(x_train, y_train)
        val_dataset = TensorDataset(x_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # 优化器和损失函数
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        model.to(self.device)
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0

        history = {'loss': [], 'val_loss': [], 'val_accuracy': []}

        for epoch in range(150):
            # 训练阶段
            model.train()
            train_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            # 验证阶段
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()

            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = 100 * correct / total

            history['loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['val_accuracy'].append(val_accuracy)

            if verbose:
                print(f'Epoch [{epoch+1}/150], Train Loss: {avg_train_loss:.4f}, '
                      f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')

            # 早停和模型保存
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), best_checkpoint)
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                if verbose:
                    print("早停触发")
                break

        # 加载最佳模型
        model.load_state_dict(torch.load(best_checkpoint))

        return model, history

    def test_model_pytorch(self, model, X_test, Y_test, batch_size, out: Path):
        """测试PyTorch模型"""

        test_dataset = TensorDataset(X_test, Y_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        model.to(self.device)
        model.eval()

        # 计算准确率
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                outputs = model(batch_x)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())

        accuracy = 100 * correct / total
        print(f"[SCORE] 准确率: {accuracy:.2f}%")

        # 计算混淆矩阵
        num_classes = len(torch.unique(Y_test))
        conf = np.zeros([num_classes, num_classes])

        for i in range(len(all_labels)):
            j = all_labels[i]
            k = all_predictions[i]
            conf[j, k] += 1

        confnorm = np.zeros([num_classes, num_classes])
        for i in range(num_classes):
            confnorm[i, :] = conf[i, :] / np.sum(conf[i, :])

        devices = [f'Device_{i}' for i in range(num_classes)]
        saveplotpath = out / "matrix.png"
        self.plot_confusion_matrix(confnorm, saveplotpath, labels=devices)

        # 分类报告
        classification_report_fp = classification_report(
            all_labels, all_predictions, target_names=devices
        )

        print(classification_report_fp)
        report_path = out / "classification_report.txt"

        with open(report_path, "w", encoding='utf-8') as file:
            file.write(classification_report_fp)

        # 保存准确率
        with open(out.joinpath("Accuracylos.txt"), "w", encoding='utf-8') as file:
            file.write(f"准确率: {accuracy:.2f}%")

        return accuracy

    def gratitude_pr(self, rank_result, n, out_dir: Path):
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
        np.savetxt(out_dir.joinpath("1-pr.csv"), pruning_rate, delimiter=",")
        return pruning_rate

    def automatic_pruner_pytorch(self, rank_path: Path, out: Path, n: int):
        """调用梯度剪枝"""
        rank_result = pd.read_csv(rank_path, header=None).values
        rr = []
        for r in rank_result:
            r = r[~np.isnan(r)]
            rr.append(r)

        # 调用gratitude_pr函数
        pruning_rates = self.gratitude_pr(rr, n, out)
        return pruning_rates

    def load_hacker_data(self):
        """加载黑客数据（简化版本）"""
        # 这里需要根据你的数据加载逻辑进行实现
        # 返回: X_train, Y_train, X_val, Y_val, X_test, Y_test
        # 暂时返回模拟数据
        print("加载模拟数据...")
        X_train = torch.randn(100, 2, 1024, 1)  # 修正维度
        Y_train = torch.randint(0, 10, (100,))
        X_val = torch.randn(20, 2, 1024, 1)
        Y_val = torch.randint(0, 10, (20,))
        X_test = torch.randn(20, 2, 1024, 1)
        Y_test = torch.randint(0, 10, (20,))

        return X_train, Y_train, X_val, Y_val, X_test, Y_test

    def prune_automatic(self, model_path: Path, output_dir: Path, prune_type: PruneType = PruneType.l2,
                       h_val: int = 10, show_model_summary: bool = True, skip_finetune: bool = False,
                       verbose: bool = True, training_verbose: bool = True):
        """
        自动剪枝主函数

        参数:
            model_path: 模型路径
            output_dir: 输出目录
            prune_type: 剪枝类型
            h_val: 剪枝参数
            show_model_summary: 是否显示模型摘要
            skip_finetune: 是否跳过微调
            verbose: 是否显示详细信息
            training_verbose: 训练时是否显示详细信息
        """

        batch_size = 32  # 减小批大小以适应小数据集

        # 1. 加载模型和数据
        print("加载模型...")
        if not model_path.exists():
            print(f"模型文件不存在: {model_path}")
            return None

        model = self.load_model(model_path)
        if show_model_summary:
            print("模型结构:")
            print(model)

        print("加载数据...")
        X_train, Y_train, X_val, Y_val, X_test, Y_test = self.load_hacker_data()

        if verbose:
            tot = X_train.shape[0] + X_val.shape[0] + X_test.shape[0]
            print(f"总IQ轨迹数: {tot}")
            print(f"训练数据形状: {X_train.shape}")
            print(f"验证数据形状: {X_val.shape}")
            print(f"测试数据形状: {X_test.shape}")

        # 2. 生成剪枝排名
        print("生成剪枝排名...")
        prune_start = datetime.now()
        self.extract_weight(model, output_dir)

        # 3. 生成剪枝文件
        print("生成剪枝文件...")
        self.automatic_pruner_pytorch(output_dir.joinpath("l2.csv"), output_dir, h_val)

        # 4. 应用自定义剪枝率创建剪枝模型
        print("创建剪枝模型...")
        custom_pruning_file = output_dir.joinpath("1-pr.csv")
        ranks_path = output_dir.joinpath("l2_idx.csv")

        if not custom_pruning_file.exists() or not ranks_path.exists():
            print("剪枝文件生成失败!")
            return None

        pruned_model = self.custom_prune_model(
            model_path, custom_pruning_file, ranks_path, X_train, Y_train
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
                pruned_model, X_train, Y_train, X_val, Y_val, output_dir,
                batch_size=batch_size, verbose=1 if training_verbose else 0
            )

            finetune_runtime = datetime.now() - finetune_start

            # 6. 测试最终模型
            print("测试模型...")
            accuracy = self.test_model_pytorch(finetuned_pruned_model, X_test, Y_test,
                                          batch_size=batch_size, out=output_dir)

            print(f"微调时间: {finetune_runtime.total_seconds()}秒")
            if history and 'loss' in history:
                print(f"微调轮数: {len(history['loss'])}")
                if len(history['loss']) > 0:
                    print(f"每轮平均时间: {finetune_runtime.total_seconds()/len(history['loss'])}秒")

            print(f"最终准确率: {accuracy:.2f}%")
        else:
            print("跳过微调步骤")

        print(f"总剪枝运行时间: {prune_runtime.total_seconds()}秒")
        return pruned_model


def create_test_model():
    """创建测试用的简单模型"""
    return TestModel()


def main():
    """主函数示例"""
    config = Config()
    pruner = PyTorchPruner(config)

    # 设置路径
    model_path = Path("test_model.pth")
    output_dir = Path("pruning_results")

    # 检查模型文件是否存在
    if not model_path.exists():
        print(f"模型文件不存在: {model_path}")
        print("创建测试模型...")

        # 创建测试模型
        test_model = create_test_model()
        # 保存完整模型，而不仅仅是state_dict
        torch.save(test_model, model_path)
        print(f"已创建测试模型: {model_path}")

    print("开始剪枝流程...")

    # 执行自动剪枝
    try:
        pruned_model = pruner.prune_automatic(
            model_path=model_path,
            output_dir=output_dir,
            prune_type=PruneType.l2,
            h_val=10,
            verbose=True,
            training_verbose=True,
            skip_finetune=True  # 跳过微调以快速测试
        )

        if pruned_model:
            print("剪枝完成!")
        else:
            print("剪枝失败!")

    except Exception as e:
        print(f"剪枝过程中出错: {e}")
        print("尝试仅生成排名文件...")

        # 仅生成排名文件
        model = pruner.load_model(model_path)
        pruner.extract_weight(model, output_dir)
        print(f"排名文件已生成到: {output_dir}")


if __name__ == "__main__":
    main()