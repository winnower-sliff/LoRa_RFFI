import os

import numpy as np

# 从配置模块导入配置、设备和模式枚举
from core.config import Config, Mode, PreprocessType, PCA_FILE_INPUT, PCA_FILE_OUTPUT, PCA_DIM_TRAIN
from modes.classification_mode import test_classification
from modes.prune_mode import pruning
from modes.rogue_device_detection_mode import test_rogue_device_detection
from modes.train_mode import train
from modes.distillation_mode import distillation, finetune_with_awgn
from training_utils.data_preprocessor import prepare_train_data
from utils.PCA import perform_pca, extract_features
from utils.better_print import print_colored_text


def main(mode=Mode.TRAIN):
    """主函数"""
    config = Config(mode)

    # 打印网络类型
    print(f"Running mode: {mode}")
    if config.mode == Mode.DISTILLATION:
        print(f"Teacher Net TYPE: {config.TEACHER_NET_TYPE}")
        print(f"Student Net TYPE: {config.STUDENT_NET_TYPE}")
    else:
        print(f"Net TYPE: {config.NET_TYPE}")

    # 使用字典映射替代 if-elif-else 结构
    mode_functions = {
        Mode.TRAIN: run_train_mode,
        Mode.CLASSIFICATION: run_classification_mode,
        Mode.ROGUE_DEVICE_DETECTION: run_rogue_device_detection_mode,
        Mode.PRUNE: run_pruning_mode,
        Mode.DISTILLATION: run_distillation_mode,
    }

    # 执行对应模式的函数
    if mode in mode_functions:
        mode_functions[mode](config)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def run_train_mode(config):
    """训练模式"""
    print_colored_text("训练模式", "32")
    print(f"Convert Type: {config.PPS_FOR}")

    data, labels = prepare_train_data(
        config.new_file_flag,
        config.filename_train_prepared_data,
        path_train_original_data="dataset/Train/dataset_training_no_aug.h5",
        dev_range=np.arange(0, 40, dtype=int),
        pkt_range=np.arange(0, 800, dtype=int),
        snr_range=np.arange(20, 80),
        generate_type=config.PROPRECESS_TYPE,
        WST_J=config.WST_J,
        WST_Q=config.WST_Q,
    )

    # 训练特征提取模型
    train(
        config.mode,
        data,
        labels,
        num_epochs=max(config.TEST_LIST),
        net_type=config.NET_TYPE,
        preprocess_type=config.PROPRECESS_TYPE,
        test_list=config.TEST_LIST,
        model_dir=config.ORIGIN_MODEL_DIR,
    )


def run_classification_mode(config):
    """分类模式"""
    print_colored_text("分类模式", "32")

    # 执行分类任务
    test_classification(
        config.mode,
        file_path_enrol="dataset/Train/dataset_training_no_aug.h5",
        file_path_clf="dataset/Test/dataset_seen_devices.h5",
        dev_range_enrol=np.arange(0, 30, dtype=int),
        pkt_range_enrol=np.arange(200, 300, dtype=int),
        dev_range_clf=np.arange(0, 30, dtype=int),
        pkt_range_clf=np.arange(0, 200, dtype=int),
        net_type=config.NET_TYPE,
        preprocess_type=config.PROPRECESS_TYPE,
        test_list=config.TEST_LIST,
        # snr_range=np.arange(20, 30),
        model_dir=config.MODEL_DIR,
        pps_for=config.PPS_FOR,
    )


def run_rogue_device_detection_mode(config):
    """甄别恶意模式"""
    print_colored_text("甄别恶意模式", "32")

    # 执行恶意设备检测任务
    test_rogue_device_detection(
        config.mode,
        file_path_enrol="dataset/Train/dataset_training_no_aug.h5",
        dev_range_enrol=np.arange(0, 30, dtype=int),
        pkt_range_enrol=np.arange(0, 100, dtype=int),
        file_path_legitimate="dataset/Test/dataset_seen_devices.h5",
        dev_range_legitimate=np.arange(0, 30, dtype=int),
        pkt_range_legitimate=np.arange(100, 200, dtype=int),
        file_path_rogue="dataset/Test/dataset_rogue.h5",
        dev_range_rogue=np.arange(40, 45, dtype=int),
        pkt_range_rogue=np.arange(0, 200, dtype=int),
        net_type=config.NET_TYPE,
        preprocess_type=config.PROPRECESS_TYPE,
        test_list=config.TEST_LIST,
        model_dir=config.MODEL_DIR,
        wst_j=config.WST_J,
        wst_q=config.WST_Q,
    )


def run_pruning_mode(config):
    """剪枝模式"""

    if config.prune_mode == 0 or config.prune_mode == 1:

        print_colored_text("剪枝模式", "32")

        data, labels = prepare_train_data(
            config.new_file_flag,
            config.filename_train_prepared_data,
            path_train_original_data="dataset/Train/dataset_training_no_aug.h5",
            dev_range=np.arange(0, 10, dtype=int),
            pkt_range=np.arange(0, 300, dtype=int),
            snr_range=np.arange(20, 80),
            generate_type=config.PROPRECESS_TYPE,
            WST_J=config.WST_J,
            WST_Q=config.WST_Q,
        )

        # 执行剪枝任务
        pruning(
            data,
            labels,
            model_dir=config.MODEL_DIR,
            config=config,
            net_type=config.NET_TYPE,
            preprocess_type=config.PROPRECESS_TYPE,
            test_list=config.TEST_LIST,
        )

    if config.prune_mode == 0 or config.prune_mode == 2:

        # 测试最终模型
        print("测试模型...")
        print_colored_text("剪枝后的分类模式", "32")

        # 执行分类任务
        test_classification(
            config.mode,
            file_path_enrol="dataset/Train/dataset_training_no_aug.h5",
            file_path_clf="dataset/Test/dataset_seen_devices.h5",
            dev_range_enrol=np.arange(0, 20, dtype=int),
            pkt_range_enrol=np.arange(0, 200, dtype=int),
            dev_range_clf=np.arange(0, 20, dtype=int),
            pkt_range_clf=np.arange(0, 200, dtype=int),
            net_type=config.NET_TYPE,
            preprocess_type=config.PROPRECESS_TYPE,
            test_list=config.TEST_LIST,
            model_dir=config.MODEL_DIR,
            pps_for=config.PPS_FOR,
        )


def run_distillation_mode(config):
    """蒸馏模式"""

    if config.DISTILLATE_MODE == 0 or config.DISTILLATE_MODE == 1:

        print_colored_text("蒸馏模式", "32")

        # 准备训练数据
        data, labels = prepare_train_data(
            config.new_file_flag,
            config.filename_train_prepared_data,
            path_train_original_data="dataset/Train/dataset_training_no_aug.h5",
            dev_range=np.arange(0, 40, dtype=int),
            pkt_range=np.arange(0, 800, dtype=int),
            snr_range=np.arange(20, 80),
            generate_type=config.PROPRECESS_TYPE,
            WST_J=config.WST_J,
            WST_Q=config.WST_Q,
        )

        # 提取教师模型特征
        if config.IS_PCA_TRAIN and not os.path.exists(PCA_FILE_INPUT):
            extract_features(data, labels, batch_size=128,
                             model_path=config.TEACHER_MODEL_DIR + "origin/Extractor_200.pth",  # 默认使用第200轮的模型
                             output_path=PCA_FILE_INPUT,
                             teacher_net_type=config.TEACHER_NET_TYPE, preprocess_type=PreprocessType.STFT
                             )
            print("PCA extract done.")
        # 执行PCA
        if config.IS_PCA_TRAIN and not os.path.exists(PCA_FILE_OUTPUT):
            perform_pca(input_file=PCA_FILE_INPUT, output_file=PCA_FILE_OUTPUT, n_components=PCA_DIM_TRAIN)
            print("PCA done.")

        # 执行蒸馏训练
        student_model = distillation(
            data,
            labels,
            teacher_model_path=config.TEACHER_MODEL_DIR + "origin/Extractor_200.pth",  # 默认使用第200轮的模型
            num_epochs=max(config.TEST_LIST),
            temperature=3.0,
            alpha=0.7,
            teacher_net_type=config.TEACHER_NET_TYPE,
            student_net_type=config.STUDENT_NET_TYPE,
            preprocess_type=config.PROPRECESS_TYPE,
            test_list=config.TEST_LIST,
            model_dir_path=config.STUDENT_MODEL_DIR + "distillation/",
            is_pca=config.IS_PCA_TRAIN,
        )

    if config.DISTILLATE_MODE == 2:

        # 微调阶段 - 使用不同SNR级别的噪声数据
        # snr_levels = [[30, 40], [20, 30], [10, 20], [0, 10]]  # 可以定义多个SNR范围
        snr_levels = [[20, 30], [10, 20], [0, 10]]  # 可以定义多个SNR范围

        for snr_range in snr_levels:
            finetuned_model = finetune_with_awgn(
                data=data,
                labels=labels,
                pretrained_model_path=config.STUDENT_MODEL_DIR + "distillation/Extractor_best1.pth",
                snr_range=snr_range,
                net_type=config.STUDENT_NET_TYPE,
                preprocess_type=config.PROPRECESS_TYPE,
                test_list=[10, 20, 30, 50],
                model_dir_path=config.STUDENT_MODEL_DIR + "distillation/"
            )

    if config.DISTILLATE_MODE == 0 or config.DISTILLATE_MODE == 3:

        # 测试最终模型
        print("测试模型...")
        print_colored_text("蒸馏后的分类模式", "32")

        # 执行分类测试
        test_classification(
            config.mode,
            file_path_enrol="dataset/Train/dataset_training_no_aug.h5",
            file_path_clf="dataset/Test/dataset_seen_devices.h5 ",
            dev_range_enrol=np.arange(0, 30, dtype=int),
            pkt_range_enrol=np.arange(400, 450, dtype=int),
            dev_range_clf=np.arange(0, 30, dtype=int),
            pkt_range_clf=np.arange(0, 400, dtype=int),
            net_type=config.STUDENT_NET_TYPE,
            preprocess_type=config.PROPRECESS_TYPE,
            test_list=config.TEST_LIST,
            # snr_range=np.arange(10, 20),
            model_dir=config.STUDENT_MODEL_DIR,
            pps_for=config.PPS_FOR,
            is_pac=config.IS_PCA_TEST,
        )
    if config.DISTILLATE_MODE == 0 or config.DISTILLATE_MODE == 4:

        print_colored_text("蒸馏后的甄别恶意模式", "32")

        # 执行恶意设备检测任务
        test_rogue_device_detection(
            config.mode,
            file_path_enrol="dataset/Train/dataset_training_no_aug.h5",
            dev_range_enrol=np.arange(0, 30, dtype=int),
            pkt_range_enrol=np.arange(0, 200, dtype=int),
            file_path_legitimate="dataset/Test/dataset_seen_devices.h5",
            dev_range_legitimate=np.arange(0, 30, dtype=int),
            pkt_range_legitimate=np.arange(100, 200, dtype=int),
            file_path_rogue="dataset/Test/dataset_rogue.h5",
            dev_range_rogue=np.arange(40, 45, dtype=int),
            pkt_range_rogue=np.arange(0, 200, dtype=int),
            net_type=config.NET_TYPE,
            preprocess_type=config.PROPRECESS_TYPE,
            test_list=config.TEST_LIST,
            model_dir=config.MODEL_DIR,
            wst_j=config.WST_J,
            wst_q=config.WST_Q,
        )
