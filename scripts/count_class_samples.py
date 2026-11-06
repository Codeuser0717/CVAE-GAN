import os
import sys
import torch
import numpy as np

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import datasets, utils

# 确保数据集已加载
if len(datasets.tr_samples) == 0:
    print("数据集未加载，请先运行相关的训练脚本或预处理脚本")
else:
    # 统计训练集中每个类别的样本数量
    tr_labels = datasets.tr_labels.cpu().numpy()
    tr_classes, tr_counts = np.unique(tr_labels, return_counts=True)
    
    # 统计测试集中每个类别的样本数量
    te_labels = datasets.te_labels.cpu().numpy()
    te_classes, te_counts = np.unique(te_labels, return_counts=True)
    
    # 确保所有类别都在结果中显示
    all_classes = sorted(set(tr_classes.tolist() + te_classes.tolist()))
    
    print("===== 数据集类别分布统计 =====")
    print(f"总类别数: {len(all_classes)}")
    print(f"训练集总样本数: {len(tr_labels)}")
    print(f"测试集总样本数: {len(te_labels)}")
    
    print("\n训练集类别分布:")
    for cls in all_classes:
        count = tr_counts[np.where(tr_classes == cls)][0] if cls in tr_classes else 0
        percentage = (count / len(tr_labels)) * 100 if len(tr_labels) > 0 else 0
        print(f"类别 {cls}: {count} 个样本 ({percentage:.2f}%)")
    
    print("\n测试集类别分布:")
    for cls in all_classes:
        count = te_counts[np.where(te_classes == cls)][0] if cls in te_classes else 0
        percentage = (count / len(te_labels)) * 100 if len(te_labels) > 0 else 0
        print(f"类别 {cls}: {count} 个样本 ({percentage:.2f}%)")