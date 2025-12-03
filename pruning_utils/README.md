# Pruning Utils 模型剪枝工具

该文件夹包含了模型剪枝相关的工具和实用程序。

## 文件说明

- [Pytorch_prunner.py](Pytorch_prunner.py)：基于PyTorch框架的模型剪枝器实现
- [TinyML_prunner.py](TinyML_prunner.py)：针对TinyML设备优化的剪枝器实现
- [rank_gen.py](rank_gen.py)：剪枝过程中权重排名生成器，用于确定哪些参数应该被剪枝

这些工具提供了对神经网络模型进行剪枝的能力，从而减小模型尺寸并提高推理速度，特别适用于资源受限的环境。