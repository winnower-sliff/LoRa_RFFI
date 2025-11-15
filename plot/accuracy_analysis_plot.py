import os
import yaml
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any

def load_experiment_records(log_dir: str) -> List[Dict[Any, Any]]:
    """
    加载所有实验记录

    Args:
        log_dir: 实验日志目录路径

    Returns:
        包含所有实验记录的列表
    """
    records = []
    for filename in os.listdir(log_dir):
        if filename.endswith('.yaml'):
            filepath = os.listdir(log_dir)
            filepath = os.path.join(log_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                record = yaml.safe_load(f)
                records.append(record)
    return records

def extract_detailed_accuracy_data(records: List[Dict[Any, Any]]) -> Dict[str, Dict[str, Dict[int, float]]]:
    """
    从实验记录中提取详细的准确率数据（按算法和epoch分类）

    Args:
        records: 实验记录列表

    Returns:
        三层嵌套字典：{实验ID: {算法名: {epoch: 准确率}}}
    """
    detailed_data = {}
    for record in records:
        exp_id = record.get("experiment_id", "Unknown")
        exp_data = {}
        config = record.get("config", {})
        exp_data["mode"] = config.get("mode", "Unknown") if isinstance(config, dict) else "Unknown"
        exp_data["model"] = config.get("model", "Unknown") if isinstance(config, dict) else "Unknown"

        if "results" in record and "classification" in record["results"]:
            classification_data = record["results"]["classification"]

            # 收集所有算法在所有epoch的数据
            for epoch_key, epoch_data in classification_data.items():
                if epoch_key.startswith("epoch_") and "accuracies" in epoch_data:
                    try:
                        epoch_num = int(epoch_key.split("_")[1])
                        for algo_name, accuracy in epoch_data["accuracies"].items():
                            if algo_name not in exp_data:
                                exp_data[algo_name] = {}
                            exp_data[algo_name][epoch_num] = accuracy
                    except (ValueError, IndexError):
                        continue

        detailed_data[exp_id] = exp_data
    return detailed_data

def plot_detailed_accuracy_comparison(detailed_data: Dict[str, Dict[str, Dict[int, float]]], save_path: str = None):
    """
    绘制详细的准确率对比图（按算法分类）

    Args:
        detailed_data: 详细的准确率数据
        save_path: 图片保存路径（可选）
    """
    # 设置中文字体
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 确定需要绘制的算法类别
    all_algorithms = set()
    for exp_data in detailed_data.values():
        all_algorithms.update(exp_data.keys())

    algorithm_groups = {
        'SVM无投票': [algo for algo in all_algorithms if 'svm_wo_voting' in algo.lower()],
        'SVM有投票': [algo for algo in all_algorithms if 'svm_w_voting' in algo.lower()],
        'KNN无投票': [algo for algo in all_algorithms if 'knn_wo_voting' in algo.lower()],
        'KNN有投票': [algo for algo in all_algorithms if 'knn_w_voting' in algo.lower()],
        '联合投票': [algo for algo in all_algorithms if 'combined' in algo.lower()],
    }

    # 过滤掉空的组
    algorithm_groups = {k: v for k, v in algorithm_groups.items() if v}

    # 创建子图
    n_groups = len(algorithm_groups)
    fig, axes = plt.subplots(n_groups, 1, figsize=(15, 5*n_groups))
    if n_groups == 1:
        axes = [axes]

    # fig.suptitle('LoRa RFFI 分类准确率详细对比分析', fontsize=16, fontweight='bold', y=0.98)

    # 为每个算法组创建子图
    for idx, (group_name, algorithms) in enumerate(algorithm_groups.items()):
        ax = axes[idx]

        # 为每个实验绘制曲线
        for exp_id, exp_data in detailed_data.items():
            short_id = exp_id[-8:] if len(exp_id) > 8 else exp_id

            # 为每种算法绘制曲线
            for algo in algorithms:
                if algo in exp_data:
                    epochs = sorted(exp_data[algo].keys())
                    accuracies = [exp_data[algo][epoch] for epoch in epochs]

                    ax.plot(epochs, accuracies, marker='o', linewidth=2, markersize=6, label=f'{short_id}-{exp_data["mode"]}-{exp_data["model"]}')

        ax.set_title(f'{group_name}准确率变化')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('准确率')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.subplots_adjust(top=0.93)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"详细对比图表已保存至: {save_path}")

    plt.show()

def plot_epoch_comparison(records: List[Dict[Any, Any]], save_path: str = None):
    """
    绘制epoch级别的准确率对比图

    Args:
        records: 实验记录列表
        save_path: 图片保存路径（可选）
    """
    detailed_data = extract_detailed_accuracy_data(records)
    plot_detailed_accuracy_comparison(detailed_data, save_path)


def main():
    """
    主函数 - 使用示例
    """
    # 设置日志目录
    log_dir = "../experiments"  # 修改为你的实际日志目录

    # 加载实验记录
    records = load_experiment_records(log_dir)

    if not records:
        print("未找到实验记录文件")
        return

    print(f"成功加载 {len(records)} 个实验记录")

    # 提取详细准确率数据
    detailed_data = extract_detailed_accuracy_data(records)
    print("详细准确率数据:", detailed_data)

    # 绘制详细对比图
    plot_epoch_comparison(records, save_path="../experiments/detailed_accuracy_comparison.png")



if __name__ == "__main__":
    main()
