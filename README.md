# LoRa_RFFI 项目根目录

该目录包含了轻量级LoRa射频指纹识别项目的主要入口文件和其他核心组件。

## 文件说明

- [main.py](main.py)：项目主入口文件，从此处启动不同模式的任务

## 目录结构

```
LoRa_RFFI/
├── core/                           # 核心配置和控制器
│   ├── config.py                   # 项目配置管理模块
│   └── controller.py               # 主控制器，协调不同模式执行
├── modes/                          # 不同运行模式实现
│   ├── train_mode.py               # 训练模式相关函数
│   ├── classification_mode.py      # 分类模式相关函数
│   ├── rogue_device_detection_mode.py # 恶意设备检测模式
│   └── distillation_mode.py        # 知识蒸馏模式相关函数
├── net/                            # 神经网络模型定义
├── paper/                          # 论文分析和可视化工具
├── plot/                           # 数据可视化功能
├── training_utils/                 # 训练相关工具
└── utils/                          # 通用工具函数
    ├── FLOPs.py                    # 模型复杂度计算工具
    ├── PCA.py                      # 主成分分析工具
    ├── better_print.py             # 增强打印功能
```

## 使用方法

通过修改 [main.py](main.py) 文件中的 `main()` 函数参数来选择不同的运行模式，然后直接运行：

```
python main.py
```

# 数据集路径、模型保存根路径

项目使用 [paths.py](file://F:\MyDocument\A竞赛科研项目\LoRa_RFFI\paths.py) 文件统一管理所有数据集路径、模型保存路径和其他文件路径，确保路径配置的一致性和可维护性。

- 数据集路径管理
- 模型存放根路径管理
- 论文结果路径

# 核心配置说明

项目的核心配置在 `core/config.py` 文件中定义，包含了项目的全局设置和枚举类型：

### 设备配置
- `DEVICE`: 自动检测并配置计算设备，优先使用CUDA GPU，否则回退到CPU

### 运行模式枚举(Mode)
项目支持五种运行模式：
- `Mode.TRAIN`: 基础训练模式，用于训练基础模型
- `Mode.CLASSIFICATION`: 分类模式，用于设备指纹分类任务
- `Mode.ROGUE_DEVICE_DETECTION`: 恶意设备检测模式，用于检测非法设备
- `Mode.DISTILLATION`: 蒸馏模式，用于知识蒸馏训练轻量级模型

### 网络类型枚举(NetworkType)
支持多种网络架构：
- `NetworkType.RESNET`: 残差网络
- `NetworkType.DRSN`: 深度残差网络
- `NetworkType.MobileNetV1/V2`: MobileNet系列轻量级网络
- `NetworkType.LightNet`: 改进的轻量级网络

### 预处理类型枚举(PreprocessType)
支持三种数据预处理方式：
- `PreprocessType.IQ`: IQ数据直接使用，2个通道（I和Q）
- `PreprocessType.STFT`: 短时傅里叶变换，1个通道（幅度）
- `PreprocessType.WST`: 小波散射变换，2个通道（实部和虚部）

