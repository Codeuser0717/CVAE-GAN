import torch
from torch import nn
from src import config
from src.utils import init_weights


class VAEEncoderModel(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 128):
        """
        编码器模型：将输入数据映射到潜在空间
        
        Args:
            input_dim: 输入特征维度
            latent_dim: 潜在空间维度
        """
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # 动态设计编码器网络
        hidden_size1 = max(256, input_dim)
        hidden_size2 = max(128, input_dim // 2)
        hidden_size3 = max(64, input_dim // 4)
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_size1),
            nn.BatchNorm1d(hidden_size1),
            nn.LeakyReLU(0.2),
            
            nn.Linear(hidden_size1, hidden_size2),
            nn.BatchNorm1d(hidden_size2),
            nn.LeakyReLU(0.2),
            
            nn.Linear(hidden_size2, hidden_size3),
            nn.BatchNorm1d(hidden_size3),
            nn.LeakyReLU(0.2),
        )
        
        # 输出潜在分布的均值和方差
        self.fc_mu = nn.Linear(hidden_size3, latent_dim)
        self.fc_logvar = nn.Linear(hidden_size3, latent_dim)
        
        self.apply(init_weights)
    
    def forward(self, x: torch.Tensor) -> tuple:
        """
        前向传播
        
        Args:
            x: 输入数据 [batch_size, input_dim]
            
        Returns:
            mu: 潜在分布均值 [batch_size, latent_dim]
            log_var: 潜在分布对数方差 [batch_size, latent_dim]
        """
        # 验证输入形状
        if x.dim() != 2:
            raise ValueError(f"输入数据应为2D张量，实际: {x.shape}")
        if x.size(1) != self.input_dim:
            raise ValueError(f"输入特征维度不匹配，期望: {self.input_dim}，实际: {x.size(1)}")
        
        # 编码器前向传播
        hidden = self.encoder(x)
        
        # 计算均值和方差
        mu = self.fc_mu(hidden)
        log_var = self.fc_logvar(hidden)
        
        return mu, log_var
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        重参数化技巧
        
        Args:
            mu: 均值
            log_var: 对数方差
            
        Returns:
            z: 重参数化后的潜在向量
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        编码输入数据到潜在空间
        
        Args:
            x: 输入数据
            
        Returns:
            z: 潜在向量
        """
        mu, log_var = self.forward(x)
        return self.reparameterize(mu, log_var)


class VAEDecoderModel(nn.Module):
    def __init__(self, latent_dim: int, output_dim: int):
        """
        解码器模型：从潜在向量生成数据
        
        Args:
            latent_dim: 潜在空间维度
            output_dim: 输出特征维度
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        
        # 动态设计解码器网络
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
        
        # 解码器前向传播
        x = self.main_model(z)
        self.hidden_status = x
        x = self.last_layer(x)
        
        return x.view(-1, self.output_dim)
    
    def reconstruct(self, x: torch.Tensor, encoder: nn.Module) -> torch.Tensor:
        """
        重构输入数据（需要编码器）
        
        Args:
            x: 输入数据 [batch_size, input_dim]
            encoder: 编码器实例
            
        Returns:
            重构的数据 [batch_size, output_dim]
        """
        with torch.no_grad():
            # 编码到潜在空间
            z = encoder.encode(x)
            # 从潜在空间解码
            return self.forward(z)


class VAEClassifierModel(nn.Module):
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