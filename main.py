"""项目主入口文件"""
from core.config import set_seed
from core.controller import main, Mode


if __name__ == "__main__":
    # 设置随机种子确保可重现性
    set_seed(42)

    # 可以通过修改这里的参数来选择运行模式:
    # Mode.TRAIN, Mode.CLASSIFICATION, Mode.ROGUE_DEVICE_DETECTION, Mode.PRUNE, Mode.DISTILLATION
    main(Mode.DISTILLATION)

