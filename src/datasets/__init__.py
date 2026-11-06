import torch
import pandas as pd
import numpy as np

from src.config import path_config
from src.datasets.tr_dataset import TrDataset
from src.datasets.te_dataset import TeDataset

# 设置数据集路径
dataset_dir = path_config.datasets / 'CAN_HCRL_OTIDS'

# 加载CSV数据并转换为PyTorch张量
def load_csv_data(file_path):
    """加载CSV数据并转换为PyTorch张量"""
    # 设置low_memory=False以避免混合类型警告
    data = pd.read_csv(file_path, header=None, low_memory=False)
    # 尝试将所有列转换为数值类型，无法转换的将被设为NaN
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    # 填充NaN值
    data = data.fillna(0)
    return torch.tensor(data.values, dtype=torch.float32)

# 加载预处理后的CICIDS2017数据集
try:
    # 加载训练数据
    tr_samples = load_csv_data(dataset_dir / 'x_train.csv')
    tr_labels = load_csv_data(dataset_dir / 'y_train.csv')
    # 将one-hot编码的标签转换为类别索引
    tr_labels = torch.argmax(tr_labels, dim=1)
    
    # 加载测试数据
    te_samples = load_csv_data(dataset_dir / 'x_test.csv')
    te_labels = load_csv_data(dataset_dir / 'y_test.csv')
    # 将one-hot编码的标签转换为类别索引
    te_labels = torch.argmax(te_labels, dim=1)
    
    # 设置特征数量和标签数量
    feature_num = tr_samples.shape[1]
    label_num = len(torch.unique(tr_labels))
    
    print(f"成功加载CAN_HCRL_OTIDS数据集:")
    print(f"  训练集样本数: {len(tr_samples)}")
    print(f"  测试集样本数: {len(te_samples)}")
    print(f"  特征数量: {feature_num}")
    print(f"  类别数量: {label_num}")
except Exception as e:
    print(f"加载CAN_HCRL_OTIDS数据集时出错: {e}")
    print("请确保已运行sample_can_hcrl_otids.py脚本生成采样后的数据集")
    # 创建空张量作为默认值，避免程序崩溃
    tr_samples = torch.zeros(0, 0)
    tr_labels = torch.zeros(0, dtype=torch.long)
    te_samples = torch.zeros(0, 0)
    te_labels = torch.zeros(0, dtype=torch.long)
    feature_num = 0
    label_num = 0
