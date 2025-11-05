"""
Author: Mabon Manoj
自动剪枝工具 - PyTorch版本
可直接在Python中调用，无需命令行参数
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Union
import torch
import os


class AutoPruner:
    """自动剪枝器类"""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def gratitude_pr(self, rank_result: List[np.ndarray], n: int, output_dir: Path) -> List[float]:
        """
        基于梯度的剪枝算法

        参数:
            rank_result: 排名结果列表
            n: 窗口大小参数
            output_dir: 输出目录

        返回:
            pruning_rate: 剪枝率列表
        """
        gratitude = []  # 存储梯度值
        pruning_rate = []  # 存储剪枝率
        idxs = []  # 存储索引

        if self.verbose:
            print(f"开始梯度剪枝计算，窗口大小 n={n}")

        # 确保输出目录存在
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, rank in enumerate(rank_result):
            if len(rank) <= 1:
                pruning_rate.append(0.0)  # 空层或单元素层，不剪枝
                if self.verbose:
                    print(f"层 {i}: 元素过少，跳过剪枝")
                continue

            rank_data = rank[1:] if len(rank) > 1 else rank  # 跳过第一个元素
            gra = []

            # 计算每个位置的梯度
            for idx, r in enumerate(rank_data):
                if idx >= len(rank_data) - n:
                    break
                try:
                    # 计算梯度: (rank[idx + n] - r) / n
                    g = (rank_data[idx + n] - r) / n
                    gra.append(g)
                except IndexError:
                    break  # 处理索引越界

            if gra:  # 确保梯度列表不为空
                gratitude.append(np.array(gra))
            else:
                gratitude.append(np.array([0]))  # 默认梯度

        # 为每个层找到最佳剪枝点
        for i, gra in enumerate(gratitude):
            found = False
            if len(gra) > 0:
                max_gradient = max(gra)
                for idx, g in enumerate(gra):
                    if g == max_gradient:  # 找到最大梯度点
                        prune_point = int(idx + n / 2)
                        idxs.append(prune_point)
                        # 计算剪枝率: 1 - (剪枝点位置 / 总长度)
                        rate = 1 - (prune_point / len(gra))
                        pruning_rate.append(float("{:.2f}".format(rate)))
                        found = True

                        if self.verbose:
                            print(f"层 {i}: 剪枝点 {prune_point}, 剪枝率 {rate:.2f}")
                        break

            # 如果没有找到合适的点，使用默认值
            if not found:
                default_point = n // 2
                idxs.append(default_point)
                default_rate = 0.5  # 默认剪枝率
                pruning_rate.append(float("{:.2f}".format(default_rate)))

                if self.verbose:
                    print(f"层 {i}: 使用默认剪枝率 {default_rate:.2f}")

        # 剪枝率范围限制
        for i in range(len(pruning_rate)):
            if pruning_rate[i] > 0.9:
                pruning_rate[i] = 0.9  # 最大剪枝率90%
            elif pruning_rate[i] < 0:
                pruning_rate[i] = 0    # 最小剪枝率0%

        if self.verbose:
            print(f"最终剪枝索引: {idxs}")
            print(f"最终剪枝率: {pruning_rate}")

        # 保存剪枝率到CSV文件
        pruning_rate_array = np.array(pruning_rate)
        output_path = output_dir.joinpath("1-pr.csv")

        # 确保目录存在后再保存
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(output_path, pruning_rate_array, delimiter=',')

        if self.verbose:
            print(f"剪枝率已保存到: {output_path}")
        return pruning_rate

    def predefined_pr(self, rank_result: List[np.ndarray]) -> List[float]:
        """
        基于预定义阈值的剪枝算法

        参数:
            rank_result: 排名结果列表

        返回:
            arch: 剪枝率列表
        """
        arch = []  # 存储架构剪枝率

        if self.verbose:
            print("开始预定义阈值剪枝计算")

        for i, r in enumerate(rank_result):
            if len(r) <= 1:
                arch.append(0.0)  # 空层或单元素层，不剪枝
                if self.verbose:
                    print(f"层 {i}: 元素过少，跳过剪枝")
                continue

            # 设置阈值为第一个元素的80%
            threshold = r[1] * 0.8 if len(r) > 1 else r[0] * 0.8
            # 找到大于阈值的元素
            t = np.where(r > threshold)
            # 计算剪枝率: 保留的比例
            rate = len(t[0]) / len(r)
            formatted_rate = float("{:.2f}".format(rate))
            arch.append(formatted_rate)

            if self.verbose:
                print(f"层 {i}: 阈值 {threshold:.2f}, 剪枝率 {formatted_rate:.2f}")

        if self.verbose:
            print(f"预定义剪枝架构: {arch}")
        return arch

    def auto_prune(self, rank_path: Union[str, Path], output_dir: Union[str, Path],
                   n: int = 10, method: str = 'gratitude') -> Optional[List[float]]:
        """
        自动剪枝主函数

        参数:
            rank_path: 排名文件路径
            output_dir: 输出目录
            n: 梯度剪枝的窗口大小参数
            method: 剪枝方法 ('gratitude' 或 'predefined')

        返回:
            pruning_rates: 剪枝率列表，失败时返回None
        """
        # 转换路径类型
        rank_path = Path(rank_path)
        output_dir = Path(output_dir)

        # 创建输出目录
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.verbose:
            print(f"开始自动剪枝...")
            print(f"排名文件: {rank_path}")
            print(f"输出目录: {output_dir}")
            print(f"剪枝方法: {method}")
            print(f"窗口大小: {n}")

        # 检查排名文件是否存在
        if not rank_path.exists():
            print(f"错误: 排名文件不存在: {rank_path}")
            return None

        # 读取排名结果
        try:
            rank_result = pd.read_csv(rank_path, header=None).values
            if self.verbose:
                print(f"成功加载排名文件: {rank_path}")
                print(f"读取到 {len(rank_result)} 层数据")
        except Exception as e:
            print(f"加载排名文件失败: {e}")
            return None

        # 清理数据（移除NaN值）
        cleaned_ranks = []
        for i, r in enumerate(rank_result):
            r_clean = r[~np.isnan(r)]
            if len(r_clean) > 0:
                cleaned_ranks.append(r_clean)
                if self.verbose and i < 3:  # 只显示前3层的信息
                    print(f"层 {i}: 有效元素 {len(r_clean)} 个")
            else:
                if self.verbose:
                    print(f"层 {i}: 无有效数据，跳过")

        if self.verbose:
            print(f"处理了 {len(cleaned_ranks)} 个有效层")

        # 根据选择的方法执行剪枝
        if method == 'gratitude':
            pruning_rates = self.gratitude_pr(cleaned_ranks, n, output_dir)
        elif method == 'predefined':
            pruning_rates = self.predefined_pr(cleaned_ranks)
            # 保存预定义方法的剪枝率
            pruning_rate_array = np.array(pruning_rates)
            output_path = output_dir.joinpath("1-pr.csv")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(output_path, pruning_rate_array, delimiter=',')
        else:
            print(f"未知的剪枝方法: {method}")
            return None

        if self.verbose:
            print("剪枝完成!")
            print(f"输出目录: {output_dir}")

        # 显示统计信息
        if pruning_rates:
            avg_rate = np.mean(pruning_rates)
            max_rate = np.max(pruning_rates)
            min_rate = np.min(pruning_rates)

            if self.verbose:
                print(f"平均剪枝率: {avg_rate:.2f}")
                print(f"最大剪枝率: {max_rate:.2f}")
                print(f"最小剪枝率: {min_rate:.2f}")
                print(f"总参数量减少: {avg_rate*100:.1f}%")

        return pruning_rates

    def load_rank_data(self, rank_path: Union[str, Path]) -> Optional[List[np.ndarray]]:
        """
        加载排名数据

        参数:
            rank_path: 排名文件路径

        返回:
            cleaned_ranks: 清理后的排名数据
        """
        rank_path = Path(rank_path)

        if not rank_path.exists():
            print(f"错误: 排名文件不存在: {rank_path}")
            return None

        try:
            rank_result = pd.read_csv(rank_path, header=None).values
            print(f"成功加载排名文件: {rank_path}")

            # 清理数据
            cleaned_ranks = []
            for r in rank_result:
                r_clean = r[~np.isnan(r)]
                if len(r_clean) > 0:
                    cleaned_ranks.append(r_clean)

            print(f"加载了 {len(cleaned_ranks)} 个有效层")
            return cleaned_ranks

        except Exception as e:
            print(f"加载排名文件失败: {e}")
            return None

    def create_mock_rank_data(self, layers: int = 3, filters_per_layer: int = 10) -> List[np.ndarray]:
        """
        创建模拟排名数据用于测试

        参数:
            layers: 层数
            filters_per_layer: 每层滤波器数量

        返回:
            mock_ranks: 模拟排名数据
        """
        mock_ranks = []
        for i in range(layers):
            # 创建递减的排名数据，模拟滤波器重要性
            start_rank = 95 - i * 5  # 每层起始排名递减
            ranks = np.linspace(start_rank, 50, filters_per_layer, dtype=int)
            mock_ranks.append(ranks)

        if self.verbose:
            print(f"创建了 {layers} 层模拟数据，每层 {filters_per_layer} 个滤波器")
            for i, ranks in enumerate(mock_ranks):
                print(f"层 {i}: {ranks}")

        return mock_ranks


# 使用示例和测试函数
def demo_auto_pruner():
    """演示自动剪枝器的使用"""
    print("=== AutoPruner 演示 ===")

    # 创建剪枝器实例
    pruner = AutoPruner(verbose=True)

    # 示例1: 使用模拟数据测试
    print("\n1. 使用模拟数据测试:")
    mock_ranks = pruner.create_mock_rank_data(layers=3, filters_per_layer=10)

    output_dir = Path("demo_output")
    rates = pruner.gratitude_pr(mock_ranks, n=5, output_dir=output_dir)
    print(f"模拟数据剪枝率: {rates}")

    # 示例2: 测试预定义方法
    print("\n2. 测试预定义阈值剪枝:")
    rates_predefined = pruner.predefined_pr(mock_ranks)
    print(f"预定义方法剪枝率: {rates_predefined}")

    # 示例3: 如果存在排名文件，进行实际剪枝
    print("\n3. 尝试实际文件剪枝:")
    rank_file = Path("l2.csv")
    if rank_file.exists():
        rates = pruner.auto_prune(
            rank_path=rank_file,
            output_dir=Path("pruning_results"),
            n=10,
            method='gratitude'
        )
        if rates:
            print(f"实际文件剪枝率: {rates}")
    else:
        print(f"排名文件 {rank_file} 不存在，跳过实际测试")
        print("提示: 请先运行 ranker_prunner.py 生成 l2.csv 文件")

    print("\n=== 演示结束 ===")


def quick_test():
    """快速测试函数"""
    print("=== AutoPruner 快速测试 ===")

    pruner = AutoPruner(verbose=True)

    # 创建测试数据
    test_data = [
        np.array([95, 92, 88, 85, 80, 75, 70, 65, 60, 55]),
        np.array([90, 85, 80, 75, 70, 65, 60, 55, 50, 45]),
    ]

    # 测试梯度剪枝
    output_dir = Path("test_output")
    rates = pruner.gratitude_pr(test_data, n=5, output_dir=output_dir)

    print(f"测试完成! 剪枝率: {rates}")
    print(f"结果保存在: {output_dir}")

    # 清理测试文件
    try:
        output_file = output_dir / "1-pr.csv"
        if output_file.exists():
            os.remove(output_file)
        if output_dir.exists():
            os.rmdir(output_dir)
        print("测试文件已清理")
    except:
        print("测试文件清理失败，可手动删除 test_output 目录")

    print("=== 快速测试结束 ===")


def main():
    """主函数 - 可以直接运行测试"""
    import sys

    if len(sys.argv) == 1:
        # 没有命令行参数，运行演示
        print("选择运行模式:")
        print("1 - 完整演示")
        print("2 - 快速测试")

        try:
            choice = input("请输入选择 (1 或 2, 默认1): ").strip()
            if choice == "2":
                quick_test()
            else:
                demo_auto_pruner()
        except KeyboardInterrupt:
            print("\n用户中断")
        except Exception as e:
            print(f"运行错误: {e}")
            print("运行快速测试...")
            quick_test()
    else:
        # 有命令行参数，使用传统方式
        import argparse

        parser = argparse.ArgumentParser(description='PyTorch自动剪枝工具')
        parser.add_argument('-i', '--rank_path', type=str, required=True,
                           help='排名结果文件路径')
        parser.add_argument('-o', '--output', type=str, required=True,
                           help='输出目录')
        parser.add_argument('-N', '--n', type=int, default=10,
                           help='梯度剪枝的窗口大小参数 (默认: 10)')
        parser.add_argument('--method', choices=['gratitude', 'predefined'],
                           default='gratitude',
                           help='剪枝方法 (默认: gratitude)')
        parser.add_argument('--verbose', action='store_true',
                           help='显示详细信息')

        args = parser.parse_args()

        pruner = AutoPruner(verbose=args.verbose)
        pruning_rates = pruner.auto_prune(
            rank_path=args.rank_path,
            output_dir=args.output,
            n=args.n,
            method=args.method
        )

        if pruning_rates:
            print("剪枝率计算完成!")
            # 保存详细报告
            report_path = Path(args.output).joinpath("pruning_report.txt")
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("剪枝报告\n")
                f.write("=" * 50 + "\n")
                f.write(f"排名文件: {args.rank_path}\n")
                f.write(f"剪枝方法: {args.method}\n")
                f.write(f"窗口大小: {args.n}\n")
                f.write(f"剪枝率: {pruning_rates}\n")
                f.write(f"平均剪枝率: {np.mean(pruning_rates):.2f}\n")
                f.write(f"最大剪枝率: {np.max(pruning_rates):.2f}\n")
                f.write(f"最小剪枝率: {np.min(pruning_rates):.2f}\n")
            print(f"详细报告已保存到: {report_path}")
        else:
            print("剪枝失败!")


if __name__ == '__main__':
    main()