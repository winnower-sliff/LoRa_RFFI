"""项目主入口文件"""
from core.config import set_seed, Mode
from core.controller import main


if __name__ == "__main__":
    # 设置随机种子确保可重现性
    set_seed(42)

    # 通过修改这里的参数来选择运行模式:
    # 执行训练模式
    # main(Mode.TRAIN)
    
    # 执行分类模式
    main(Mode.CLASSIFICATION)
    
    # 执行恶意设备检测模式
    # main(Mode.ROGUE_DEVICE_DETECTION)
    
    # 执行模型剪枝模式
    # main(Mode.PRUNE)
    
    # 执行知识蒸馏模式
    # main(Mode.DISTILLATION)