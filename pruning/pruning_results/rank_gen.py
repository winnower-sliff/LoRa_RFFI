"""
Author: Mabon Manoj
排名生成工具 - PyTorch版本
用于生成模型的滤波器排名，支持L2范数和FPGM方法
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import distance
from scipy.stats import entropy

# 导入你的现有模块
try:
    from core.config import Config, DEVICE
except ImportError:
    # 如果导入失败，创建模拟配置
    class Config:
        def __init__(self):
            self.MODEL_DIR_PATH = "models"

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 在模块级别定义演示模型类
class DemoModel(nn.Module):
    """演示用模型"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc = nn.Linear(64 * 8 * 8, 10)
        self.pool = nn.AdaptiveAvgPool2d((8, 8))

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class RankGenerator:
    """排名生成器类"""

    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.device = DEVICE

    def mapping(self, W: torch.Tensor, min_w: float, max_w: float) -> torch.Tensor:
        """
        权重映射函数

        参数:
            W: 输入权重张量
            min_w: 最小值
            max_w: 最大值

        返回:
            q_w: 量化后的权重
        """
        scale_w = (max_w - min_w) / 100
        min_arr = torch.full(W.shape, min_w, device=W.device)
        q_w = torch.round((W - min_arr) / scale_w).to(torch.uint8)
        return q_w

    def calc_MI(self, x: np.ndarray, y: np.ndarray, bins: int) -> float:
        """
        计算两个数组之间的互信息

        参数:
            x: 第一个数组
            y: 第二个数组
            bins: 直方图bin数量

        返回:
            mi: 互信息值
        """
        c_xy = np.histogram2d(x, y, bins)[0]
        mi = entropy(c_xy.flatten())
        return mi

    def grouped_rank(self, feature_map: np.ndarray, num_groups: int) -> int:
        """
        分组排名计算

        参数:
            feature_map: 特征图
            num_groups: 分组数量

        返回:
            r: 矩阵秩
        """
        dis = 256 / num_groups
        grouped_feature = np.round(feature_map / dis)
        r = np.linalg.matrix_rank(grouped_feature)
        return r

    def update_dis(self, distances: Dict, layer_idx: int, dis: Dict) -> Dict:
        """
        更新层间距离

        参数:
            distances: 距离字典
            layer_idx: 层索引
            dis: 新的距离

        返回:
            更新后的距离字典
        """
        if layer_idx in distances.keys():
            for k, v in dis.items():
                distances[layer_idx][k] += v
        else:
            distances[layer_idx] = dis
        return distances

    def extract_layers(self, model: nn.Module) -> List[nn.Module]:
        """
        提取模型中的特定层

        参数:
            model: PyTorch模型

        返回:
            layers: 层列表
        """
        layers = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                layers.append(module)
        return layers

    def cal_rank(self, features: List[torch.Tensor], results: List[np.ndarray]) -> List[np.ndarray]:
        """
        计算每层滤波器的排名

        参数:
            features: 特征列表
            results: 结果列表

        返回:
            更新后的结果列表
        """
        for layer_idx, feature_layer in enumerate(features):
            after = feature_layer.squeeze()
            n_filters = after.shape[-1] if after.dim() > 1 else 1

            filter_rank = []
            if after.dim() == 2:
                for i in range(n_filters):
                    a = after[:, i]
                    rtf = torch.mean(a).item()
                    filter_rank.append(rtf)
                filter_rank = sorted(filter_rank, reverse=True)
            else:
                # 对于1D特征，直接排序
                filter_rank = sorted(after.cpu().numpy(), reverse=True)

            filter_rank = self.mapping(
                torch.tensor(filter_rank),
                min(filter_rank),
                max(filter_rank)
            ).cpu().numpy()

            if layer_idx < len(results):
                results[layer_idx] = np.add(results[layer_idx], np.array(filter_rank))
            else:
                results.append(np.array(filter_rank))

        return results

    def extract_feature_maps(self, model: nn.Module, input_data: torch.Tensor,
                           num_trace: int = 50) -> List[np.ndarray]:
        """
        提取特征图

        参数:
            model: PyTorch模型
            input_data: 输入数据
            num_trace: 跟踪数量

        返回:
            R_list: 排名列表
        """
        # 获取所有卷积层和全连接层
        feature_layers = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                feature_layers.append(module)

        results = [np.zeros(1) for _ in feature_layers]  # 初始化结果

        model.eval()
        with torch.no_grad():
            for i in range(min(num_trace, len(input_data))):
                x = input_data[i].unsqueeze(0)  # 添加batch维度

                # 逐层提取特征
                features = []
                current_output = x
                for layer in feature_layers:
                    current_output = layer(current_output)
                    features.append(current_output)

                results = self.cal_rank(features, results)

        # 计算平均排名
        R_after = np.array(results) / num_trace
        R_list = [list(r) for r in R_after]

        return R_list

    def extract_weights_l2(self, model: nn.Module, output_dir: Path) -> Tuple[List, List]:
        """
        基于L2范数提取权重排名

        参数:
            model: PyTorch模型
            output_dir: 输出目录

        返回:
            results: 排名结果
            idx_results: 索引结果
        """
        results = []
        idx_results = []

        print("开始基于L2范数的权重提取...")

        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                # 跳过分类层
                if "classification" in name:
                    continue

                print(f"处理层: {name}")
                weight = module.weight.data

                # 处理不同维度的权重
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

                # 计算L2范数
                r = torch.norm(a, dim=1)
                r = self.mapping(r, torch.min(r), torch.max(r))

                # 排序并保存结果
                sorted_r = sorted(r.cpu().numpy(), reverse=True)
                results.append(sorted_r)

                idx_dis = torch.argsort(r, dim=0).cpu().numpy()
                idx_results.append(idx_dis)

                print(f"  - 滤波器数量: {len(sorted_r)}")
                print(f"  - 排名范围: {min(sorted_r)} ~ {max(sorted_r)}")

        # 保存结果
        output_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(results, index=None)
        df.to_csv(output_dir / "l2.csv", header=False, index=False)

        df_idx = pd.DataFrame(idx_results, index=None)
        df_idx.to_csv(output_dir / "l2_idx.csv", header=False, index=False)

        print(f"L2排名文件已保存到: {output_dir}")
        return results, idx_results

    def extract_weights_fpgm(self, model: nn.Module, output_dir: Path, dist_type: str = "l2") -> Tuple[List, List]:
        """
        基于FPGM(Filter Pruning via Geometric Median)提取权重排名

        参数:
            model: PyTorch模型
            output_dir: 输出目录
            dist_type: 距离类型 ('l2', 'l1', 'cos')

        返回:
            results: 排名结果
            idx_results: 索引结果
        """
        results = []
        idx_results = []

        print(f"开始基于FPGM的权重提取 (距离类型: {dist_type})...")

        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                print(f"处理层: {name}")
                weight = module.weight.data

                # 重塑权重向量
                if weight.ndim == 4:  # Conv2D
                    weight_vec = weight.view(weight.size(0), -1).cpu().numpy().T
                elif weight.ndim == 3:  # Conv1D
                    weight_vec = weight.view(weight.size(0), -1).cpu().numpy().T
                elif weight.ndim == 2:  # Linear
                    weight_vec = weight.cpu().numpy().T
                else:
                    continue

                # 计算距离矩阵
                if dist_type == "l2" or dist_type == "l1":
                    dist_matrix = distance.cdist(weight_vec, weight_vec, 'euclidean')
                elif dist_type == "cos":
                    dist_matrix = 1 - distance.cdist(weight_vec, weight_vec, 'cosine')
                else:
                    raise ValueError(f"不支持的距离类型: {dist_type}")

                # 计算挤压矩阵
                squeeze_matrix = np.sum(np.abs(dist_matrix), axis=0)
                distance_sum = sorted(squeeze_matrix, reverse=True)

                # 映射和排序
                idx_dis = np.argsort(squeeze_matrix, axis=0)
                r = self.mapping(
                    torch.tensor(distance_sum),
                    min(distance_sum),
                    max(distance_sum)
                ).cpu().numpy()

                results.append(r)
                idx_results.append(idx_dis)

                print(f"  - 滤波器数量: {len(r)}")
                print(f"  - 距离范围: {min(distance_sum):.4f} ~ {max(distance_sum):.4f}")

        # 保存结果
        output_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(results, index=None)
        df.to_csv(output_dir / "fpgm.csv", header=False, index=False)

        df_idx = pd.DataFrame(idx_results, index=None)
        df_idx.to_csv(output_dir / "fpgm_idx.csv", header=False, index=False)

        print(f"FPGM排名文件已保存到: {output_dir}")
        return results, idx_results

    def load_model(self, model_path: Path) -> nn.Module:
        """
        加载PyTorch模型

        参数:
            model_path: 模型路径

        返回:
            model: 加载的模型
        """
        try:
            # 尝试加载完整模型
            model = torch.load(model_path, map_location=self.device, weights_only=False)
            if isinstance(model, nn.Module):
                return model
            else:
                # 如果是state_dict，需要知道模型结构
                raise ValueError("需要提供模型类来加载state_dict")
        except Exception as e:
            print(f"加载模型失败: {e}")
            raise

    def generate_ranks(self, model_path: Path, output_dir: Path, method: str = "l2",
                      dist_type: str = "l2") -> bool:
        """
        生成排名的主函数

        参数:
            model_path: 模型路径
            output_dir: 输出目录
            method: 方法类型 ('l2', 'fpgm')
            dist_type: 距离类型 (仅用于FPGM)

        返回:
            success: 是否成功
        """
        try:
            # 加载模型
            model = self.load_model(model_path)
            print(f"模型加载成功: {model_path}")

            # 显示模型信息
            total_params = sum(p.numel() for p in model.parameters())
            print(f"模型总参数量: {total_params:,}")

            # 根据方法生成排名
            if method == "l2":
                results, idx_results = self.extract_weights_l2(model, output_dir)
            elif method == "fpgm":
                results, idx_results = self.extract_weights_fpgm(model, output_dir, dist_type)
            else:
                raise ValueError(f"不支持的方法: {method}")

            # 显示统计信息
            print(f"\n=== 排名生成统计 ===")
            print(f"处理层数: {len(results)}")
            for i, (result, idx_result) in enumerate(zip(results, idx_results)):
                print(f"层 {i}: {len(result)} 个滤波器, 排名范围: {min(result)} ~ {max(result)}")

            return True

        except Exception as e:
            print(f"排名生成失败: {e}")
            return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='PyTorch排名生成工具')
    parser.add_argument('-i', '--model_dir', type=str, required=True,
                       help='模型文件路径')
    parser.add_argument('-o', '--output', type=str, default='rank_results',
                       help='输出目录')
    parser.add_argument('-type', '--type', choices={'l2', 'fpgm'}, default='l2',
                       help='排名方法类型')
    parser.add_argument('--dist_type', choices={'l2', 'l1', 'cos'}, default='l2',
                       help='距离类型 (仅用于FPGM方法)')
    parser.add_argument('--config', type=str, help='配置文件路径')

    args = parser.parse_args()

    # 加载配置
    if args.config:
        config = Config()
    else:
        config = Config()

    # 创建排名生成器
    rank_gen = RankGenerator(config)

    # 生成排名
    success = rank_gen.generate_ranks(
        model_path=Path(args.model_dir),
        output_dir=Path(args.output),
        method=args.type,
        dist_type=args.dist_type
    )

    if success:
        print(f"\n✓ 排名生成完成!")
        print(f"输出目录: {args.output}")

        # 显示生成的文件
        output_path = Path(args.output)
        files = list(output_path.glob("*.csv"))
        if files:
            print("生成的文件:")
            for file in files:
                print(f"  - {file.name}")
    else:
        print("\n✗ 排名生成失败!")
        sys.exit(1)


# 集成到现有框架的便捷函数
def generate_model_ranks(model_path: Union[str, Path], output_dir: Union[str, Path] = "rank_results",
                        method: str = "l2", config: Config = None) -> bool:
    """
    便捷函数：生成模型排名

    参数:
        model_path: 模型路径
        output_dir: 输出目录
        method: 排名方法
        config: 配置对象

    返回:
        success: 是否成功
    """
    rank_gen = RankGenerator(config)
    return rank_gen.generate_ranks(
        model_path=Path(model_path),
        output_dir=Path(output_dir),
        method=method
    )


def create_simple_test_model():
    """创建简单的测试模型"""
    class SimpleTestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 8, 3)  # 简单卷积层
            self.conv2 = nn.Conv2d(8, 16, 3)
            self.fc = nn.Linear(16 * 6 * 6, 10)  # 假设输入为 28x28

        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x

    return SimpleTestModel()


def demo_rank_generation():
    """演示排名生成功能"""
    print("=== 排名生成演示 ===")

    # 创建和保存测试模型
    model = create_simple_test_model()
    model_path = Path("demo_model.pth")

    # 保存模型状态字典而不是完整模型
    torch.save(model.state_dict(), model_path)
    print(f"创建演示模型: {model_path}")

    # 由于我们保存的是state_dict，需要创建新的模型实例来加载
    loaded_model = create_simple_test_model()
    loaded_model.load_state_dict(torch.load(model_path))
    torch.save(loaded_model, model_path)  # 现在保存完整模型

    # 生成排名
    rank_gen = RankGenerator()

    print("\n1. 使用L2方法生成排名...")
    success_l2 = rank_gen.generate_ranks(
        model_path=model_path,
        output_dir=Path("demo_l2_results"),
        method="l2"
    )

    if success_l2:
        print("\n2. 使用FPGM方法生成排名...")
        success_fpgm = rank_gen.generate_ranks(
            model_path=model_path,
            output_dir=Path("demo_fpgm_results"),
            method="fpgm"
        )
    else:
        success_fpgm = False

    # 清理
    if model_path.exists():
        model_path.unlink()

    # 清理演示目录
    import shutil
    for demo_dir in ["demo_l2_results", "demo_fpgm_results"]:
        if Path(demo_dir).exists():
            shutil.rmtree(demo_dir)

    if success_l2 and success_fpgm:
        print("\n✓ 演示完成!")
        return True
    else:
        print("\n✗ 演示失败!")
        return False


def quick_demo():
    """快速演示"""
    print("=== 快速演示排名生成 ===")

    # 直接使用DemoModel类
    model = DemoModel()
    model_path = Path("quick_demo_model.pth")

    # 保存模型
    torch.save(model, model_path)
    print(f"创建演示模型: {model_path}")

    # 生成排名
    rank_gen = RankGenerator()

    success = rank_gen.generate_ranks(
        model_path=model_path,
        output_dir=Path("quick_demo_results"),
        method="l2"
    )

    # 清理
    if model_path.exists():
        model_path.unlink()

    import shutil
    if Path("quick_demo_results").exists():
        shutil.rmtree("quick_demo_results")

    if success:
        print("✓ 快速演示完成!")
    else:
        print("✗ 快速演示失败!")

    return success


if __name__ == '__main__':
    # 如果没有命令行参数，运行快速演示
    if len(sys.argv) == 1:
        print("未提供命令行参数，运行快速演示...")
        quick_demo()
    else:
        main()