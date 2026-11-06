import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm
from src import config
from src.utils import init_weights


class SNGANGeneratorModel(nn.Module):
    def __init__(self, latent_dim: int, num_classes: int, output_dim: int):
        """
        谱归一化生成器模型：根据潜在向量和条件生成数据
        
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
        
        # SNGAN的生成器通常不使用谱归一化，使用批归一化
        self.main_model = nn.Sequential(
            spectral_norm(nn.Linear(total_input_dim, hidden_size1)),
            nn.BatchNorm1d(hidden_size1),
            nn.ReLU(),  # SNGAN通常使用ReLU而不是LeakyReLU
            
            spectral_norm(nn.Linear(hidden_size1, hidden_size2)),
            nn.BatchNorm1d(hidden_size2),
            nn.ReLU(),
            
            spectral_norm(nn.Linear(hidden_size2, hidden_size3)),
            nn.BatchNorm1d(hidden_size3),
            nn.ReLU(),
        )
        
        self.hidden_status: torch.Tensor = None
        
        # 输出层 - 使用Sigmoid匹配[0,1]数据范围
        self.last_layer = nn.Sequential(
            spectral_norm(nn.Linear(hidden_size3, output_dim)),
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


class SNGANDiscriminatorModel(nn.Module):
    """
    谱归一化判别器网络：使用谱归一化稳定训练
    """
    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        
        # 输入维度 = 特征维度 + 条件维度（one-hot编码）
        total_input_dim = in_features + num_classes
        
        # 专门为判别任务设计的网络，全部使用谱归一化
        hidden_size1 = max(256, total_input_dim)
        hidden_size2 = max(128, total_input_dim // 2)
        hidden_size3 = 64
        
        # SNGAN判别器：所有层都使用谱归一化，使用LeakyReLU激活
        self.discriminator_network = nn.Sequential(
            spectral_norm(nn.Linear(total_input_dim, hidden_size1)),
            nn.LeakyReLU(0.2),
            
            spectral_norm(nn.Linear(hidden_size1, hidden_size2)),
            nn.LeakyReLU(0.2),
            
            spectral_norm(nn.Linear(hidden_size2, hidden_size3)),
            nn.LeakyReLU(0.2),
            
            # 输出层也使用谱归一化
            spectral_norm(nn.Linear(hidden_size3, 1)),
            # 不使用激活函数，使用hinge损失
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
        前向传播 - 输出判别结果
        
        Args:
            x: 输入数据 [batch_size, in_features]
            condition: 条件信息（标签索引）
            
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


class SNGANClassifierModel(nn.Module):
    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        
        hidden_size1 = max(256, in_features)
        hidden_size2 = max(128, in_features // 2)
        hidden_size3 = 64
        
        # 分类器也可以使用谱归一化增强稳定性
        self.classifier_network = nn.Sequential(
            spectral_norm(nn.Linear(in_features, hidden_size1)),
            nn.ReLU(),
            
            spectral_norm(nn.Linear(hidden_size1, hidden_size2)),
            nn.ReLU(),
            
            spectral_norm(nn.Linear(hidden_size2, hidden_size3)),
            nn.ReLU(),
            
            # 输出层不使用激活函数
            spectral_norm(nn.Linear(hidden_size3, num_classes)),
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