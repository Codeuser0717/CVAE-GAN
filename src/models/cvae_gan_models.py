import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm
from src import config
from src.utils import init_weights

class CVAEGANEncoderModel(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, latent_dim: int = 128):
        """
        编码器模型：将输入数据和条件映射到潜在空间
        
        Args:
            input_dim: 输入特征维度
            num_classes: 类别数量（用于条件编码）
            latent_dim: 潜在空间维度
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        
        # 输入维度 = 特征维度 + 条件维度（one-hot编码）
        total_input_dim = input_dim + num_classes
        
        # 动态设计编码器网络
        hidden_size1 = max(256, total_input_dim)
        hidden_size2 = max(128, total_input_dim // 2)
        hidden_size3 = max(64, total_input_dim // 4)
        
        self.encoder = nn.Sequential(
            nn.Linear(total_input_dim, hidden_size1),
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
    
    def _process_condition(self, condition: torch.Tensor) -> torch.Tensor:
        """
        处理条件输入：将标签索引转换为one-hot编码
        
        Args:
            condition: 标签索引 [batch_size] 或 [batch_size, 1]
            
        Returns:
            one-hot编码的条件 [batch_size, num_classes]
        """
        if condition.dim() == 1:
            condition = condition.long()
        elif condition.dim() == 2 and condition.size(1) == 1:
            condition = condition.squeeze(1).long()
        else:
            raise ValueError(f"条件输入格式错误，期望1D或2D(单列)，实际: {condition.shape}")
        
        # 转换为one-hot编码
        return torch.nn.functional.one_hot(condition, num_classes=self.num_classes).float()
    
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> tuple:
        """
        前向传播
        
        Args:
            x: 输入数据 [batch_size, input_dim]
            condition: 条件信息（标签索引）[batch_size] 或 [batch_size, 1]
            
        Returns:
            mu: 潜在分布均值 [batch_size, latent_dim]
            log_var: 潜在分布对数方差 [batch_size, latent_dim]
        """
        # 验证输入形状
        if x.dim() != 2:
            raise ValueError(f"输入数据应为2D张量，实际: {x.shape}")
        if x.size(1) != self.input_dim:
            raise ValueError(f"输入特征维度不匹配，期望: {self.input_dim}，实际: {x.size(1)}")
        
        # 处理条件输入
        condition_onehot = self._process_condition(condition)
        
        # 拼接输入数据和条件
        x_cond = torch.cat([x, condition_onehot], dim=1)
        
        # 编码器前向传播
        hidden = self.encoder(x_cond)
        
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
    
    def encode(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        编码输入数据到潜在空间
        
        Args:
            x: 输入数据
            condition: 条件信息（标签索引）
            
        Returns:
            z: 潜在向量
        """
        mu, log_var = self.forward(x, condition)
        return self.reparameterize(mu, log_var)


class CVAEGANGeneratorModel(nn.Module):
    def __init__(self, latent_dim: int, num_classes: int, output_dim: int):
        """
        条件生成器模型：根据潜在向量和条件生成数据
        
        Args:
            latent_dim: 潜在空间维度
            num_classes: 类别数量（用于条件编码）
            output_dim: 输出特征维度
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.output_dim = output_dim
        
        # 输入维度 = 潜在维度 + 条件维度（one-hot编码）
        total_input_dim = latent_dim + num_classes
        
        # 动态设计生成器网络
        hidden_size1 = max(256, total_input_dim)
        hidden_size2 = max(128, total_input_dim // 2)
        hidden_size3 = max(64, total_input_dim // 4)
        
        self.main_model = nn.Sequential(
            nn.Linear(total_input_dim, hidden_size1),
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
    
    def _process_condition(self, condition: torch.Tensor, target_batch_size: int = None) -> torch.Tensor:
        """
        处理条件输入：将标签索引转换为one-hot编码
        
        Args:
            condition: 标签索引，支持多种格式
            target_batch_size: 目标batch大小（用于广播）
            
        Returns:
            one-hot编码的条件
        """
        # 处理标量条件（单个类别）
        if condition.dim() == 0:
            condition = condition.unsqueeze(0)
        
        # 处理不同格式的输入
        if condition.dim() == 1:
            # [batch_size] 格式
            condition = condition.long()
            if target_batch_size and condition.size(0) == 1:
                # 广播单个条件到整个batch
                condition = condition.repeat(target_batch_size)
        elif condition.dim() == 2 and condition.size(1) == 1:
            # [batch_size, 1] 格式
            condition = condition.squeeze(1).long()
        else:
            raise ValueError(f"条件输入格式错误: {condition.shape}")
        
        # 转换为one-hot编码
        return torch.nn.functional.one_hot(condition, num_classes=self.num_classes).float()

    def generate_conditional_samples(self, num: int, condition: torch.Tensor) -> torch.Tensor:
        """
        生成指定条件的样本
        
        Args:
            num: 生成样本数量
            condition: 条件信息（标签索引），支持标量、向量等格式
            
        Returns:
            生成的样本 [num, output_dim]
        """
        # 处理条件输入
        condition_onehot = self._process_condition(condition, target_batch_size=num)
        
        # 验证条件形状
        if condition_onehot.size(0) != num:
            raise ValueError(f"条件数量不匹配，期望: {num}，实际: {condition_onehot.size(0)}")
        
        # 生成随机潜在向量
        z = torch.randn(num, self.latent_dim, device=config.device)
        return self.forward(z, condition_onehot)

    def forward(self, z: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            z: 潜在向量 [batch_size, latent_dim]
            condition: 条件信息（one-hot编码）[batch_size, num_classes]
            
        Returns:
            生成的样本 [batch_size, output_dim]
        """
        # 验证输入形状
        if z.dim() != 2:
            raise ValueError(f"潜在向量应为2D张量，实际: {z.shape}")
        if z.size(1) != self.latent_dim:
            raise ValueError(f"潜在维度不匹配，期望: {self.latent_dim}，实际: {z.size(1)}")
        
        if condition.dim() != 2:
            raise ValueError(f"条件应为2D张量，实际: {condition.shape}")
        if condition.size(1) != self.num_classes:
            raise ValueError(f"条件维度不匹配，期望: {self.num_classes}，实际: {condition.size(1)}")
        
        if z.size(0) != condition.size(0):
            raise ValueError(f"batch大小不匹配，潜在向量: {z.size(0)}，条件: {condition.size(0)}")
        
        # 拼接潜在向量和条件
        z_cond = torch.cat([z, condition], dim=1)
        
        # 生成器前向传播
        x = self.main_model(z_cond)
        self.hidden_status = x
        x = self.last_layer(x)
        
        return x.view(-1, self.output_dim)
    
    def reconstruct(self, x: torch.Tensor, condition: torch.Tensor, encoder: nn.Module) -> torch.Tensor:
        """
        重构输入数据（需要编码器）
        
        Args:
            x: 输入数据 [batch_size, input_dim]
            condition: 条件信息（标签索引）
            encoder: 编码器实例
            
        Returns:
            重构的数据 [batch_size, output_dim]
        """
        with torch.no_grad():
            # 编码到潜在空间
            z = encoder.encode(x, condition)
            # 处理条件为one-hot编码
            condition_onehot = self._process_condition(condition, target_batch_size=x.size(0))
            # 从潜在空间解码
            return self.forward(z, condition_onehot)


class CVAEGANDiscriminatorModel(nn.Module):
    """
    独立的判别器网络：专门用于判断输入数据的真实性
    """
    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        
        # 输入维度 = 特征维度 + 条件维度（one-hot编码）
        total_input_dim = in_features + num_classes
        
        # 专门为判别任务设计的网络
        hidden_size1 = max(256, total_input_dim)
        hidden_size2 = max(128, total_input_dim // 2)
        hidden_size3 = 64
        
        self.discriminator_network = nn.Sequential(
            spectral_norm(nn.Linear(total_input_dim, hidden_size1)),
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
    
    def _process_condition(self, condition: torch.Tensor, target_batch_size: int = None) -> torch.Tensor:
        """处理条件输入：将标签索引转换为one-hot编码"""
        if condition.dim() == 0:
            condition = condition.unsqueeze(0)
        
        if condition.dim() == 1:
            condition = condition.long()
            if target_batch_size:
                if condition.size(0) == 1:
                    # 如果只有一个条件，重复到目标batch大小
                    condition = condition.repeat(target_batch_size)
                elif condition.size(0) != target_batch_size:
                    # 如果条件batch大小与目标batch大小不匹配，抛出错误
                    raise ValueError(f"条件batch大小不匹配，期望: {target_batch_size}，实际: {condition.size(0)}")
        elif condition.dim() == 2 and condition.size(1) == 1:
            condition = condition.squeeze(1).long()
            if target_batch_size and condition.size(0) != target_batch_size:
                raise ValueError(f"条件batch大小不匹配，期望: {target_batch_size}，实际: {condition.size(0)}")
        else:
            raise ValueError(f"条件输入格式错误: {condition.shape}")
        
        return torch.nn.functional.one_hot(condition, num_classes=self.num_classes).float()

    def forward(self, x: torch.Tensor, condition: torch.Tensor = None) -> torch.Tensor:
        """
        前向传播 - 只输出判别结果
        
        Args:
            x: 输入数据 [batch_size, in_features]
            condition: 条件信息（标签索引），可选
            
        Returns:
            判别器输出 [batch_size, 1]
        """
        # 确保输入是二维张量
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        # 处理条件输入
        if condition is not None:
            condition_onehot = self._process_condition(condition, target_batch_size=x.size(0))
            x_cond = torch.cat([x, condition_onehot], dim=1)
        else:
            condition_onehot = torch.zeros(x.size(0), self.num_classes, device=x.device)
            x_cond = torch.cat([x, condition_onehot], dim=1)
        
        # 判别器前向传播
        features = self.discriminator_network[:-1](x_cond)  # 获取最后一层之前的特征
        self.hidden_status = features
        output = self.discriminator_network[-1](features)  # 最后一层
        
        return output
    
    def get_feature_importance(self, x: torch.Tensor, condition: torch.Tensor = None):
        """获取特征重要性（基于第一层权重）"""
        with torch.no_grad():
            if condition is not None:
                condition_onehot = self._process_condition(condition, target_batch_size=x.size(0))
                x_cond = torch.cat([x, condition_onehot], dim=1)
            else:
                condition_onehot = torch.zeros(x.size(0), self.num_classes, device=x.device)
                x_cond = torch.cat([x, condition_onehot], dim=1)
            
            first_layer = self.discriminator_network[0]
            if hasattr(first_layer, 'weight'):
                weights = first_layer.weight.data
                feature_importance = torch.mean(torch.abs(weights), dim=0)
                data_importance = feature_importance[:self.in_features]
                condition_importance = feature_importance[self.in_features:]
                return data_importance, condition_importance
        return None, None


class CVAEGANClassifierModel(nn.Module):
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