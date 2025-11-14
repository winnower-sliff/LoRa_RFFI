import numpy as np

# 从配置模块导入配置、设备和模式枚举
from core.config import Config, Mode
from modes.classification_mode import test_classification
from modes.prune_mode import pruning
from modes.rogue_device_detection_mode import test_rogue_device_detection
from modes.train_mode import train
from training_utils.data_preprocessor import prepare_train_data
from utils.better_print import print_colored_text


def main(mode=Mode.TRAIN):
    """主函数"""
    config = Config()

    # 移除命令行参数解析，保留默认值
    print(f"Running mode: {mode}")
    print(f"Net NAME: {config.NET_NAME}")

    # 使用字典映射替代 if-elif-else 结构
    mode_functions = {
        Mode.TRAIN: run_train_mode,
        Mode.CLASSIFICATION: run_classification_mode,
        Mode.ROGUE_DEVICE_DETECTION: run_rogue_device_detection_mode,
        Mode.PRUNING: run_pruning_mode,
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
        data,
        labels,
        num_epochs=max(config.TEST_LIST),
        net_type=config.NET_TYPE,
        preprocess_type=config.PROPRECESS_TYPE,
        test_list=config.TEST_LIST,
        model_dir_path=config.ORIGIN_MODEL_DIR,
    )


def run_classification_mode(config):
    """分类模式"""
    print_colored_text("分类模式", "32")

    # 执行分类任务
    test_classification(
        file_path_enrol="dataset/Train/dataset_training_no_aug.h5",
        file_path_clf="dataset/Test/dataset_seen_devices.h5 ",
        dev_range_enrol=np.arange(15, 30, dtype=int),
        pkt_range_enrol=np.arange(200, 400, dtype=int),
        dev_range_clf=np.arange(15, 30, dtype=int),
        pkt_range_clf=np.arange(0, 200, dtype=int),
        net_type=config.NET_TYPE,
        preprocess_type=config.PROPRECESS_TYPE,
        test_list=config.TEST_LIST,
        model_dir=config.MODEL_DIR,
        net_name=config.NET_NAME,
        pps_for=config.PPS_FOR,
    )


def run_rogue_device_detection_mode(config):
    """甄别恶意模式"""
    print_colored_text("甄别恶意模式", "32")

    # 执行恶意设备检测任务
    test_rogue_device_detection(
        file_path_enrol="./dataset/Test/dataset_residential.h5",
        dev_range_enrol=np.arange(30, 40, dtype=int),
        pkt_range_enrol=np.arange(0, 100, dtype=int),
        file_path_legitimate="./dataset/Test/dataset_residential.h5",
        dev_range_legitimate=np.arange(30, 40, dtype=int),
        pkt_range_legitimate=np.arange(100, 200, dtype=int),
        file_path_rogue="./dataset/Test/dataset_rogue.h5",
        dev_range_rogue=np.arange(40, 45, dtype=int),
        pkt_range_rogue=np.arange(0, 100, dtype=int),
        net_type=config.NET_TYPE,
        preprocess_type=config.PROPRECESS_TYPE,
        test_list=config.TEST_LIST,
        model_dir_path=config.MODEL_DIR,
        wst_j=config.WST_J,
        wst_q=config.WST_Q,
    )

def run_pruning_mode(config):
    """剪枝模式"""
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
        dev_range_enrol=np.arange(0, 20, dtype=int),
        pkt_range_enrol=np.arange(0, 200, dtype=int),
        dev_range_clf=np.arange(0, 20, dtype=int),
        pkt_range_clf=np.arange(0, 200, dtype=int),
        net_type=config.NET_TYPE,
        preprocess_type=config.PROPRECESS_TYPE,
        test_list=config.TEST_LIST,
    )