"""项目主入口文件"""
from core.config import set_seed, Mode, DistillateMode, NetworkType, PreprocessType
from core.controller import main


if __name__ == "__main__":
    # 设置随机种子确保可重现性
    set_seed(42)

    # 可以通过修改这里的参数来选择运行模式:
    # Mode.TRAIN, Mode.CLASSIFICATION, Mode.ROGUE_DEVICE_DETECTION, Mode.DISTILLATION
    # main(Mode.TRAIN, net_type=NetworkType.RESNET)

    # --- 推荐的Pipeline ---
    # # 1.训练ResNet教师模型
    # main(Mode.TRAIN, net_type=NetworkType.RESNET)
    # # 2. 知识蒸馏训练LightNet学生模型
    main(Mode.DISTILLATION,
         teacher_net_type=NetworkType.RESNET,
         student_net_type=NetworkType.LightNet,
         distillate_mode=DistillateMode.ALL)
    
    # --- 其他模式使用示例 ---
    # 1. 分类模式：使用已训练的模型对设备进行分类
    #   默认使用LightNet网络和STFT预处理，可调整net_type和preprocess_type参数
    #   main(Mode.CLASSIFICATION, net_type=NetworkType.LightNet, preprocess_type=PreprocessType.STFT)
    #
    # 2. 恶意设备检测模式：检测未知设备是否为恶意设备
    #   需要已训练的模型，默认使用ResNet网络
    #   main(Mode.ROGUE_DEVICE_DETECTION, net_type=NetworkType.RESNET, preprocess_type=PreprocessType.STFT)
    #
    # 3. 蒸馏模式的其他选项：
    #   - 仅执行蒸馏训练: DistillateMode.ONLY_DISTILLATE
    #   - 仅测试学生模型: DistillateMode.ONLY_TEST
    #   - 仅进行恶意设备检测: DistillateMode.ONLY_ROGUE
    #   示例：
    #   main(Mode.DISTILLATION,
    #        teacher_net_type=NetworkType.RESNET,
    #        student_net_type=NetworkType.LightNet,
    #        distillate_mode=DistillateMode.ONLY_TEST)
    #
    # 4. 可选参数说明：
    #   - net_type: 网络类型，可选值: NetworkType.RESNET, NetworkType.DRSN, NetworkType.MobileNetV1, NetworkType.MobileNetV2, NetworkType.LightNet
    #   - preprocess_type: 预处理类型，可选值: PreprocessType.IQ, PreprocessType.STFT, PreprocessType.WST
    #   - is_pca_train: 训练时是否使用PCA降维 (默认为True)
    #   - is_pca_test: 测试时是否使用PCA降维 (默认为True)
    #   - teacher_net_type, student_net_type: 蒸馏模式专用
    #   - distillate_mode: 蒸馏训练模式，可选值: DistillateMode.ALL, DistillateMode.ONLY_DISTILLATE, DistillateMode.ONLY_TEST, DistillateMode.ONLY_ROGUE