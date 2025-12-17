"""项目主入口文件"""
from core.config import set_seed, Mode, DistillateMode, NetworkType
from core.controller import main


if __name__ == "__main__":
    # 设置随机种子确保可重现性
    set_seed(42)

    # 可以通过修改这里的参数来选择运行模式:
    # Mode.TRAIN, Mode.CLASSIFICATION, Mode.ROGUE_DEVICE_DETECTION, Mode.DISTILLATION
    main(Mode.CLASSIFICATION)

    # --- 推荐的Pipeline ---
    # # 1.训练ResNet教师模型
    # main(Mode.TRAIN, net_type=NetworkType.RESNET)
    # # 2. 知识蒸馏训练LightNet学生模型
    # main(Mode.DISTILLATION,
    #      teacher_net_type=NetworkType.RESNET,
    #      student_net_type=NetworkType.LightNet,
    #      distillate_mode=DistillateMode.ALL)
