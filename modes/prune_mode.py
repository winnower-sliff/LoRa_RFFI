"""剪枝模式相关函数"""
import os
from datetime import datetime

import numpy as np
from sklearn.model_selection import train_test_split

from core.config import Config, NetworkType, PRUNED_OUTPUT_DIR, H_VAL
from core.config import PruneType

from modes.classification_mode import test_classification
from pruning.ranker_prunner import extract_weight, automatic_pruner_pytorch, custom_prune_model, finetune_model_pytorch
from training_utils.data_preprocessor import load_model
from utils.better_print import print_colored_text


def pruning(
    data,
    labels,
    config: Config,
    origin_model_path: str,
    pruned_model_path: str,
    preprocess_type,
    test_list: list,
    prune_type: PruneType = PruneType.l2,
    batch_size=32,
    show_model_summary=False,
    skip_finetune=False,
    verbose=True,
    training_verbose=True,
):
    """

    :param data: 训练数据。
    :param labels: 训练标签
    :param config: 配置文件
    :param origin_model_path: 获取初始模型的地址
    :param pruned_model_path: 剪枝后模型的地址
    :param preprocess_type: 预处理类型
    :param test_list: 测试点列表
    :param prune_type: 剪枝类型
    :param batch_size: 减小批大小以适应小数据集
    :param show_model_summary: 是否显示模型摘要
    :param skip_finetune: 是否跳过微调
    :param verbose: 是否显示详细信息
    :param training_verbose: 训练时是否显示详细信息
    """

    # 0 for all, 1 for only prune, 2 for only test
    prune_mode = 0

    if prune_mode == 0 or prune_mode == 1:

        # 执行剪枝
        print("开始模型剪枝...")

        for exit_epoch in test_list or []:
            print(exit_epoch, test_list)
            print()
            print("=============================")
            origin_model_dir = origin_model_path + f"Extractor_{exit_epoch}.pth"
            pruned_model_dir = pruned_model_path + f"Extractor_{exit_epoch}.pth"
            if not os.path.exists(origin_model_path):
                print(f"{origin_model_path} isn't exist")
            else:
                # 加载模型和数据
                print("\n加载模型...")

                model = load_model(origin_model_dir, NetworkType.WAVELET.value, preprocess_type)
                if show_model_summary:
                    print("模型结构:")
                    print(model)

                print("\n加载数据...")

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
                    print(f"  - 总IQ轨迹数: {tot}")
                    print(f"  - 训练数据形状: {data_train.shape}")
                    print(f"  - 验证数据形状: {data_valid.shape}")
                    print(f"  - 测试数据形状: {data_test.shape}")

                prune_ranks_path = PRUNED_OUTPUT_DIR + f"Extractor_{exit_epoch}_l2_idx.csv"
                prune_rank_path = PRUNED_OUTPUT_DIR + f"Extractor_{exit_epoch}_l2.csv"
                custom_pruning_file = os.path.join(PRUNED_OUTPUT_DIR, f"Extractor_{exit_epoch}_1-pr.csv")

                # 生成剪枝排名
                print("\n生成剪枝排名...")
                prune_start = datetime.now()
                extract_weight(model, prune_rank_path, prune_ranks_path, show_model_summary)

                # 生成剪枝文件
                print("\n生成剪枝文件...")
                automatic_pruner_pytorch(H_VAL, prune_rank_path, exit_epoch)

                # 应用自定义剪枝率创建剪枝模型
                print("\n创建剪枝模型...")

                pruned_model = custom_prune_model(
                    origin_model_dir, custom_pruning_file, prune_ranks_path, preprocess_type, show_model_summary
                )

                prune_runtime = datetime.now() - prune_start

                if verbose:
                    print(f"剪枝运行时间: {prune_runtime.total_seconds()}秒")

                if show_model_summary:
                    print("剪枝后模型结构:")
                    print(pruned_model)

                if not skip_finetune:
                    print("\n开始微调...")
                    finetune_start = datetime.now()

                    # 5. 微调剪枝模型
                    finetuned_pruned_model, history = finetune_model_pytorch(
                        pruned_model, data_train, labels_train, data_valid, labels_valid,
                        checkpoint_dir=pruned_model_dir, exit_epoch=exit_epoch,
                        batch_size=batch_size, verbose=1 if training_verbose else 0
                    )

                    finetune_runtime = datetime.now() - finetune_start

                    print(f"微调时间: {finetune_runtime.total_seconds()}秒")
                    if history and 'loss' in history:
                        print(f"微调轮数: {len(history['loss'])}")
                        if len(history['loss']) > 0:
                            print(f"每轮平均时间: {finetune_runtime.total_seconds() / len(history['loss'])}秒")

                    # print(f"最终准确率: {accuracy:.2f}%")
                else:
                    print("跳过微调步骤")

                print(f"总剪枝运行时间: {prune_runtime.total_seconds()}秒")

    if prune_mode == 0 or prune_mode == 2:
        # 测试最终模型
        print("测试模型...")

        print_colored_text("分类模式", "32")

        # 执行分类任务
        test_classification(
            file_path_enrol="dataset/Train/dataset_training_no_aug.h5",
            file_path_clf="dataset/Test/dataset_seen_devices.h5",
            dev_range_enrol=np.arange(15, 30, dtype=int),
            pkt_range_enrol=np.arange(0, 200, dtype=int),
            dev_range_clf=np.arange(15, 30, dtype=int),
            pkt_range_clf=np.arange(0, 200, dtype=int),
            net_type=config.NET_TYPE,
            preprocess_type=preprocess_type,
            test_list=config.TEST_LIST,
            model_dir_path=config.MODEL_DIR_PATH,
            wst_j=config.WST_J,
            wst_q=config.WST_Q,
            net_name=config.NET_NAME,
            pps_for=config.PPS_FOR,
            config=config
        )

    return
