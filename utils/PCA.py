# utils/PCA.py
import numpy as np
import torch
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

from core.config import Config, DEVICE, Mode
from net.TripletNet import TripletNet
from training_utils.TripletDataset import TripletDataset


def extract_features(
        data,
        labels,
        batch_size,
        model_path,
        output_path,
        teacher_net_type,
        preprocess_type,
):
    """
    使用预训练的教师模型提取数据特征并保存
    
    :param data: 输入数据
    :param labels: 数据标签
    :param batch_size: 批处理大小
    :param model_path: 预训练模型路径
    :param output_path: 特征输出保存路径
    :param teacher_net_type: 教师网络类型
    :param preprocess_type: 预处理类型配置对象，需包含in_channels属性
    :return: None
    """

    # 生成数据加载器
    dataset = TripletDataset(data, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    teacher_model = TripletNet(net_type=teacher_net_type, in_channels=preprocess_type.in_channels)  # MobileNet
    teacher_model.load_state_dict(torch.load(model_path))
    teacher_model.to(DEVICE)
    teacher_model.eval()

    feats = []
    labels_list = []

    with torch.no_grad():
        for batch_idx, (anchor, positive, negative) in enumerate(loader):
            anchor, positive, negative = (
                anchor.to(DEVICE),
                positive.to(DEVICE),
                negative.to(DEVICE),
            )
            # teacher 输出高维特征
            teacher_anchor, teacher_positive, teacher_negative = teacher_model(
                anchor, positive, negative
            )  # shape (B, D_t)

            # 保存对应 label（anchor label 就可以）
            feats.append(teacher_anchor.cpu().numpy())
            labels_list.append(labels[batch_idx * batch_size: (batch_idx + 1) * batch_size])  # 或者 anchor labels

    feats = np.concatenate(feats, axis=0)
    labels_arr = np.concatenate(labels_list, axis=0)
    np.savez(output_path, feats=feats, labels=labels_arr)
    print("Teacher features saved, shape:", feats.shape)


def perform_pca(input_file="teacher_feats.npz", output_file="pca_16.npz", n_components=16):
    """
    对输入特征进行PCA降维处理
    
    :param input_file: 输入的特征文件路径
    :param output_file: 输出的PCA参数文件路径
    :param n_components: 降维后的维度数
    :return: components, mean
    """
    
    # 加载数据
    data = np.load(input_file)
    feats = data['feats']
    labels = data['labels']
    
    # 执行PCA
    pca = PCA(n_components=n_components)
    pca.fit(feats)
    components = pca.components_  # d x D_t
    mean = pca.mean_
    
    # 保存结果
    np.savez(output_file, components=components, mean=mean)
    print("PCA done, shape:", components.shape)
    
    return components, mean

if __name__ == '__main__':
    config = Config(Mode.TEST)
    # 提取特征
    # extract_features()
    # 执行PCA
    perform_pca()
