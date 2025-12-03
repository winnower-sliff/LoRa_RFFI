# LoRa_RFFI 项目根目录

该目录包含了LoRa射频指纹识别项目的主要入口文件和其他核心组件。

## 文件说明

- [main.py](main.py)：项目主入口文件，从此处启动不同模式的任务
- [experiment_logger.py](experiment_logger.py)：实验日志记录器，用于记录实验过程和结果
- [Performance_Comparison.py](Performance_Comparison.py)：性能比较工具，用于对比不同模型或方法的性能
- [tmp_pingpian.py](tmp_pingpian.py)：临时芯片测试相关功能

## 目录结构

```
LoRa_RFFI/
├── core/                  # 核心配置和控制器
├── modes/                 # 不同的运行模式实现
├── net/                   # 神经网络模型定义
├── plot/                  # 数据可视化功能
├── pruning_utils/         # 模型剪枝工具
├── training_utils/        # 训练相关工具
├── utils/                 # 通用工具函数
├── Openset_RFFI_TIFS/     # 原始开源项目代码和数据集
```

## 使用方法

通过修改 [main.py](main.py) 文件中的 `main()` 函数参数来选择不同的运行模式：
- `Mode.TRAIN`: 基础训练模式
- `Mode.CLASSIFICATION`: 分类模式
- `Mode.ROGUE_DEVICE_DETECTION`: 恶意设备检测模式
- `Mode.PRUNE`: 模型剪枝模式
- `Mode.DISTILLATION`: 知识蒸馏模式

然后直接运行 [main.py](main.py) 启动项目。