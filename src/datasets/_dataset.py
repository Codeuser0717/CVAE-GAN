import torch
from src import datasets, config

class Dataset:
    def __init__(self, training: bool = True):
        self.training = training
        # 不再在初始化时固定引用，而是动态访问全局数据集
    
    def __len__(self):
        if self.training:
            return len(datasets.tr_labels)
        else:
            return len(datasets.te_labels)
    
    def __getitem__(self, idx: int):
        if self.training:
            # 动态访问当前最新的全局数据
            sample = datasets.tr_samples[idx].to(config.device)
            label = datasets.tr_labels[idx].to(config.device)
        else:
            sample = datasets.te_samples[idx].to(config.device)
            label = datasets.te_labels[idx].to(config.device)
        return sample, label
