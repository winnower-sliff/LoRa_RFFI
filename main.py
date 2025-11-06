"""项目主入口文件"""
from core.controller import main, Mode


if __name__ == "__main__":
    # 可以通过修改这里的参数来选择运行模式:
    # Mode.TRAIN, Mode.CLASSIFICATION, Mode.ROGUE_DEVICE_DETECTION, Mode.PRUNING
    main(Mode.PRUNING)