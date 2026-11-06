import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm
from src import config
from src.utils import init_weights


class GANGeneratorModel(nn.Module):
    def __init__(self, latent_dim: int, output_dim: int):
        """
        生成器模型：从潜在向量生成数据
        
        Args:
            latent_dim: 潜在空间维度
            output_dim: 输出特征维度
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        
        # 动态设计生成器网络
        hidden_size1 = max(256, latent_dim)
        hidden_size2 = max(128, latent_dim // 2)
        hidden_size3 = max(64, latent_dim // 4)
        
        self.main_model = nn.Sequential(
            nn.Linear(latent_dim, hidden_size1),
            nn.BatchNorm1d(hidden_size1),
            nn.LeakyReLU(0.2),
            
            nn.Linear(hidden_size1, hidden_size2),
            nn.BatchNorm1d(hidden_size2),
            nn.LeakyReLU(0.2),
            
            nn.Linear(hidden_size2, hidden_size3),
            nn.BatchNorm1d(hidden_size3),
            nn.LeakyReLU(0.2),
        )
        
        self.hidden_status: torch.Tensor = None
        
        # 输出层 - 使用Sigmoid匹配[0,1]数据范围
        self.last_layer = nn.Sequential(
            nn.Linear(hidden_size3, output_dim),
            nn.Sigmoid()  # 输出范围[0,1]，匹配预处理后的数据
        )
        
        self.apply(init_weights)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            z: 潜在向量 [batch_size, latent_dim]
            
        Returns:
            生成的样本 [batch_size, output_dim]
        """
        # 验证输入形状
        if z.dim() != 2:
            raise ValueError(f"潜在向量应为2D张量，实际: {z.shape}")
        if z.size(1) != self.latent_dim:
            raise ValueError(f"潜在维度不匹配，期望: {self.latent_dim}，实际: {z.size(1)}")
        
        # 生成器前向传播
        x = self.main_model(z)
        self.hidden_status = x
        x = self.last_layer(x)
        
        return x.view(-1, self.output_dim)


class GANDiscriminatorModel(nn.Module):
    """
    判别器网络：判断输入数据的真实性
    """
    def __init__(self, in_features: int):
        super().__init__()
        self.in_features = in_features
        
        # 专门为判别任务设计的网络
        hidden_size1 = max(256, in_features)
        hidden_size2 = max(128, in_features // 2)
        hidden_size3 = 64
        
        self.discriminator_network = nn.Sequential(
            spectral_norm(nn.Linear(in_features, hidden_size1)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            spectral_norm(nn.Linear(hidden_size1, hidden_size2)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            spectral_norm(nn.Linear(hidden_size2, hidden_size3)),
            nn.LeakyReLU(0.2),
            
            spectral_norm(nn.Linear(hidden_size3, 1)),
            # 不使用激活函数，使用Wasserstein损失
        )
        
        self.hidden_status: torch.Tensor = None
        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播 - 只输出判别结果
        
        Args:
            x: 输入数据 [batch_size, in_features]
            
        Returns:
            判别器输出 [batch_size, 1]
        """
        # 确保输入是二维张量
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        # 判别器前向传播
        features = self.discriminator_network[:-1](x)  # 获取最后一层之前的特征
        self.hidden_status = features
        output = self.discriminator_network[-1](features)  # 最后一层
        
        return output
    
    def get_feature_importance(self, x: torch.Tensor):
        """获取特征重要性（基于第一层权重）"""
        with torch.no_grad():
            first_layer = self.discriminator_network[0]
            if hasattr(first_layer, 'weight'):
                weights = first_layer.weight.data
                feature_importance = torch.mean(torch.abs(weights), dim=0)
                return feature_importance
        return None


class GANClassifierModel(nn.Module):
    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        
        hidden_size1 = max(256, in_features)
        hidden_size2 = max(128, in_features // 2)
        hidden_size3 = 64
        
        self.classifier_network = nn.Sequential(
            nn.Linear(in_features, hidden_size1),
            
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_size1, hidden_size2),
            nn.LayerNorm(hidden_size2),  # 使用LayerNorm
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_size2, hidden_size3),
            nn.ReLU(),
            
            # 移除Softmax，只保留线性层
            nn.Linear(hidden_size3, num_classes),
        )
        
        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 确保输入是二维张量
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.classifier_network(x)  # 输出logits，不要Softmax
    
    def get_feature_importance(self, x: torch.Tensor):
        """获取特征重要性（基于第一层权重）"""
        with torch.no_grad():
            first_layer = self.classifier_network[0]
            if hasattr(first_layer, 'weight'):
                weights = first_layer.weight.data
                feature_importance = torch.mean(torch.abs(weights), dim=0)
                return feature_importance
        return None