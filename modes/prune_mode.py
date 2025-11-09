"""剪枝模式相关函数"""
import numpy as np

from core.config import Config
from core.config import PruneType
from modes.classification_mode import test_classification
from pruning.ranker_prunner import PyTorchPruner
from utils.better_print import print_colored_text


def pruning(
    data,
    labels,
    config: Config,
    origin_model_path: str,
    pruned_model_path: str,
    test_list: list,
):
    """

    :param data: 训练数据。
    :param labels: 训练标签
    :param config: 配置文件
    :param origin_model_path: 获取初始模型的地址
    :param pruned_model_path: 剪枝后模型的地址
    """

    try:
        # 初始化剪枝器
        pruner = PyTorchPruner(
            config=config,
            output_dir=config.pruned_output_dir
        )

        # 执行剪枝
        print("开始模型剪枝...")
        pruner.prune_automatic(
            data,
            labels,
            origin_model_dir=origin_model_path,
            pruned_model_dir=pruned_model_path,
            test_list=test_list,
            prune_type=PruneType.l2,
            h_val=config.H_VAL,
            show_model_summary=False,
            verbose=True,
            training_verbose=True,
            skip_finetune=False
        )

    except Exception as e:
        print(f"剪枝过程中出错: {e}")
        import traceback
        traceback.print_exc()

    # 测试最终模型
    print("测试模型...")

    print_colored_text("分类模式", "32")

    # 执行分类任务
    test_classification(
        file_path_enrol="dataset/Train/dataset_training_no_aug.h5",
        file_path_clf="dataset/Test/dataset_seen_devices.h5 ",
        dev_range_enrol=np.arange(10, 20, dtype=int),
        pkt_range_enrol=np.arange(0, 200, dtype=int),
        dev_range_clf=np.arange(10, 20, dtype=int),
        pkt_range_clf=np.arange(0, 200, dtype=int),
        net_type=config.NET_TYPE,
        preprocess_type=config.PROPRECESS_TYPE,
        test_list=config.TEST_LIST,
        model_dir_path=config.MODEL_DIR_PATH,
        wst_j=config.WST_J,
        wst_q=config.WST_Q,
        net_name=config.NET_NAME,
        pps_for=config.PPS_FOR,
        config=config
    )

    return
