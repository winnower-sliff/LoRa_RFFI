# LoRa_RFFI 项目根目录

该目录包含了LoRa射频指纹识别项目的主要入口文件和其他核心组件。

## 文件说明

- [main.py](main.py)：项目主入口文件，从此处启动不同模式的任务

## 目录结构

```
LoRa_RFFI/
├── core/                           # 核心配置和控制器
│ ├── config.py                     # 项目配置文件，包含模式定义和全局常量
│ └── controller.py                 # 主控制器，协调不同模式的执行流程
├── modes/                          # 不同的运行模式实现
│ ├── train_mode.py                 # 训练模式相关函数 
│ ├── classification_mode.py        # 分类模式相关函数 
│ ├── rogue_device_detection_mode.py # 恶意设备检测模式相关函数 
│ ├── prune_mode.py                 # 模型剪枝模式相关函数 
│ └── distillation_mode.py          # 知识蒸馏模式相关函数
├── net/                            # 神经网络模型定义
├── plot/                           # 数据可视化功能
├── pruning_utils/                  # 模型剪枝工具
├── training_utils/                 # 训练相关工具
├── utils/                          # 通用工具函数
│ ├── FLOPs.py                      # 模型复杂度计算工具 
│ ├── PCA.py                        # 主成分分析工具 
│ ├── better_print.py               # 增强打印功能
│ ├── tmp_pingpian.py               # 临时芯片测试相关功能
│ ├── Performance_Comparison.py     # 性能比较工具，用于对比不同模型或方法的性能
```

## 使用方法

通过修改 [main.py](main.py) 文件中的 `main()` 函数参数来选择不同的运行模式：

- `Mode.TRAIN`: 基础训练模式
- `Mode.CLASSIFICATION`: 分类模式
- `Mode.ROGUE_DEVICE_DETECTION`: 恶意设备检测模式
- `Mode.PRUNE`: 模型剪枝模式
- `Mode.DISTILLATION`: 知识蒸馏模式

然后直接运行 [main.py](main.py) 启动项目。

# 核心配置说明

项目的核心配置在 `core/config.py` 文件中定义，包含了项目的全局设置和枚举类型：

### 设备配置
- `DEVICE`: 自动检测并配置计算设备，优先使用CUDA GPU，否则回退到CPU

### 运行模式枚举(Mode)
项目支持五种运行模式：
- `Mode.TRAIN`: 基础训练模式，用于训练基础模型
- `Mode.CLASSIFICATION`: 分类模式，用于设备指纹分类任务
- `Mode.ROGUE_DEVICE_DETECTION`: 恶意设备检测模式，用于检测非法设备
- `Mode.PRUNE`: 剪枝模式，用于模型压缩和优化
- `Mode.DISTILLATION`: 蒸馏模式，用于知识蒸馏训练轻量级模型

### 网络类型枚举(NetworkType)
支持多种网络架构：
- `NetworkType.RESNET`: 残差网络
- `NetworkType.DRSN`: 深度残差网络
- `NetworkType.MobileNetV1/V2`: MobileNet系列轻量级网络
- `NetworkType.LightNetV1/V2`: 改进的轻量级网络

### 预处理类型枚举(PreprocessType)
支持三种数据预处理方式：
- `PreprocessType.IQ`: IQ数据直接使用，2个通道（I和Q）
- `PreprocessType.STFT`: 短时傅里叶变换，1个通道（幅度）
- `PreprocessType.WST`: 小波散射变换，2个通道（实部和虚部）

### 其他重要配置
- PCA相关配置：包括训练和测试时的PCA维度设置
- 剪枝相关配置：包括剪枝粒度和文件存储路径
- Config类：用于存储全局配置参数，包括网络类型、预处理类型、训练参数等
