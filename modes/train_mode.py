"""训练模式相关函数"""
import math
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import time

from training_utils.TripletDataset import TripletDataset, TripletLoss
from net.TripletNet import TripletNet

# 从配置模块导入设备
from core.config import DEVICE


def train(data, labels, batch_size=32, num_epochs=200, learning_rate=1e-3, 
          net_type=None, preprocess_type=None, test_list=None, model_dir_path=None):
    """
    准备数据并训练三元组网络模型。

    过程:
    1. 将数据集划分为训练集和验证集(尽管此函数只使用了训练集)。
    2. 创建三元组数据集(TripletDataset)和数据加载器(DataLoader)。
    3. 初始化三元组网络模型(TripletNet)、优化器(如Adam)和损失函数(如TripletLoss)。
    4. 将模型移动到指定的设备(如GPU)上, 并设置模型为训练模式。
    5. 在每个epoch中, 遍历数据加载器, 进行前向传播、计算损失、反向传播和优化步骤。
    6. 记录每个epoch的损失, 并在指定的轮次(test_list)保存模型状态字典。
    7. 在训练的最后几个轮次(test_list[-3:]), 绘制损失随epoch变化的图表并保存。

    :param data: 输入数据, 通常为图像特征向量。
    :param labels: 输入数据的标签。
    :param batch_size (int): 批处理大小, 每次迭代训练的网络输入数量。默认为32。
    :param num_epochs (int): 训练的轮数(遍历整个数据集的次数)。默认为200。
    :param learning_rate (float): 学习率, 控制优化器更新权重的步长。默认为1e-3。
    :param net_type: 网络类型
    :param preprocess_type: 预处理类型
    :param test_list: 测试点列表
    :param model_dir_path: 模型保存路径
    """

    # 数据集划分
    data_train, data_valid, labels_train, labels_valid = train_test_split(
        data, labels, test_size=0.1, shuffle=True
    )

    # 生成数据加载器
    train_dataset = TripletDataset(data_train, labels_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    batch_num = math.ceil(len(train_dataset) / batch_size)

    # 初始化模型和优化器
    model = TripletNet(net_type=net_type, in_channels=1 if preprocess_type == 0 else 2)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = TripletLoss(margin=0.1)

    # 训练模型
    model.to(DEVICE)
    model.train()

    print(
        "\n---------------------\n"
        "Num of epoch: {}\n"
        "Batch size: {}\n"
        "Num of train batch: {}\n"
        "---------------------\n".format(num_epochs, batch_size, batch_num)
    )
    loss_per_epoch = []

    # 总进度条
    with tqdm(total=num_epochs, desc="Total Progress") as total_bar:
        for epoch in range(num_epochs):
            start_time_ep = time.time()
            total_loss = 0.0
            # 每一轮训练进度条
            with tqdm(total=batch_num, desc=f"Epoch {epoch}", leave=False) as pbar:
                for batch_idx, (anchor, positive, negative) in enumerate(train_loader):
                    anchor, positive, negative = (
                        anchor.to(DEVICE),
                        positive.to(DEVICE),
                        negative.to(DEVICE),
                    )

                    # 前向传播
                    embedded_anchor, embedded_positive, embedded_negative = model(
                        anchor, positive, negative
                    )
                    loss = loss_fn(
                        embedded_anchor, embedded_positive, embedded_negative
                    )

                    # 反向传播与优化
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

                    pbar.update(1)

            end_time_ep = time.time()

            loss_ep = total_loss / len(train_loader) * 10

            text = (
                f"Epoch [{epoch+1}/{num_epochs}], "
                + f"time: {end_time_ep-start_time_ep:.2f}s, "
                + f"Loss: {loss_ep:.6f}"
            )

            tqdm.write(text)
            loss_per_epoch.append(loss_ep)

            # 保存训练好的模型
            if test_list and (epoch + 1) in test_list:
                # 创建文件夹&文件
                if not os.path.exists(model_dir_path):
                    os.makedirs(model_dir_path)
                file_name = f"Extractor_{epoch + 1}.pth"

                # 保存模型到指定路径
                file_path = model_dir_path + file_name
                torch.save(model.state_dict(), file_path)
                tqdm.write(f"Model saved to {file_path}")

                # 绘制loss折线图
                if test_list and (epoch + 1) in test_list[-3:]:
                    # print("Plotting results... ")
                    fig, ax1 = plt.subplots()
                    ax1.plot(
                        range(len(loss_per_epoch)),
                        loss_per_epoch,
                        label="Loss",
                        color="red",
                    )
                    ax1.set_xlabel("Epoch")
                    ax1.set_ylabel("Loss", color="red")
                    ax1.tick_params(axis="y", labelcolor="red")

                    # 添加标题和图例
                    net_name = "origin" if net_type == 0 else "drsn"
                    pps_for = "stft" if preprocess_type == 0 else "wst"
                    plt.title(
                        f"Loss of {num_epochs} Epoch, Net: {net_name}, Convert Type: {pps_for}"
                    )
                    fig.legend(
                        loc="upper right",
                        bbox_to_anchor=(1, 1),
                        bbox_transform=ax1.transAxes,
                    )

                    # 显示图表
                    plt.grid(True)

                    pic_save_path = model_dir_path + f"loss_{epoch+1}.png"
                    plt.savefig(pic_save_path)
                    # plt.show()

            # 更新总进度条
            total_bar.update(1)
    return model