# Modes 训练模式模块

该文件夹包含了项目中不同的训练和处理模式实现。

## 文件说明

- [classification_mode.py](classification_mode.py)：设备分类模式的实现，用于常规的设备识别和分类任务
- [distillation_mode.py](distillation_mode.py)：知识蒸馏模式，通过大模型指导小模型训练，提高模型效率
- [prune_mode.py](prune_mode.py)：模型剪枝模式，用于减少模型大小和计算复杂度
- [rogue_device_detection_mode.py](rogue_device_detection_mode.py)：异常设备检测模式，专门用于检测未授权或恶意设备
- [train_mode.py](train_mode.py)：基础训练模式，提供通用的模型训练功能

这些模块定义了项目可以执行的不同任务类型，每种模式对应一种特定的应用场景。