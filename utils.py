import numpy as np

import torch

from net import TripletNet


# 模型保存函数
def save_model(model, file_path="Extractor.pth"):
    """保存模型到指定路径"""
    torch.save(model.state_dict(), file_path)
    print(f"Model saved to {file_path}")


# 模型加载函数
def load_model(file_path="Extractor.pth"):
    """从指定路径加载模型"""
    model = TripletNet()
    model.load_state_dict(torch.load(file_path))
    model.eval()
    print(f"Model loaded from {file_path}")
    return model

