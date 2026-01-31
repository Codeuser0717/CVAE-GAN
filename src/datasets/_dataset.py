import torch
from src import datasets, config

class Dataset:
    def __init__(self, training: bool = True):
        self.training = training
       
    
    def __len__(self):
        if self.training:
            return len(datasets.tr_labels)
        else:
            return len(datasets.te_labels)
    
    def __getitem__(self, idx: int):
        if self.training:
         
            sample = datasets.tr_samples[idx].to(config.device)
            label = datasets.tr_labels[idx].to(config.device)
        else:
            sample = datasets.te_samples[idx].to(config.device)
            label = datasets.te_labels[idx].to(config.device)
        return sample, label
