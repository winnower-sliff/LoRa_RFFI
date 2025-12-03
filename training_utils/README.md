# Training Utils 训练工具

该文件夹包含了模型训练过程中使用的各种实用工具和辅助功能。

## 文件说明

- [TripletDataset.py](TripletDataset.py)：三元组数据集处理类，用于构建三元组训练样本
- [data_loader.py](data_loader.py)：数据加载器，负责从存储中读取和预处理数据
- [data_preprocessor.py](data_preprocessor.py)：数据预处理器，对原始数据进行清洗、转换和标准化
- [model_utils.py](model_utils.py)：模型相关工具函数集合

这些工具为模型训练提供了数据处理、加载和预处理的支持，确保训练过程能够高效地获取和处理所需数据。