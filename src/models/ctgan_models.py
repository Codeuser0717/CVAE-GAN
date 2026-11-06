import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm
from src import config
from src.utils import init_weights


class ResidualBlock(nn.Module):
    """残差块，用于CTGAN生成器和判别器"""
    def __init__(self, input_dim: int, output_dim: int, use_spectral_norm: bool = False):
        super().__init__()
        
        # 主路径
        layers = []
        if use_spectral_norm:
            layers.append(spectral_norm(nn.Linear(input_dim, output_dim)))
        else:
            layers.append(nn.Linear(input_dim, output_dim))
        layers.append(nn.BatchNorm1d(output_dim))
        layers.append(nn.ReLU())
        
        if use_spectral_norm:
            layers.append(spectral_norm(nn.Linear(output_dim, output_dim)))
        else:
            layers.append(nn.Linear(output_dim, output_dim))
        layers.append(nn.BatchNorm1d(output_dim))
        
        self.main = nn.Sequential(*layers)
        
        # 捷径连接
        self.shortcut = nn.Sequential()
        if input_dim != output_dim:
            if use_spectral_norm:
                self.shortcut = nn.Sequential(spectral_norm(nn.Linear(input_dim, output_dim)))
            else:
                self.shortcut = nn.Sequential(nn.Linear(input_dim, output_dim))
        
    def forward(self, x):
        return self.main(x) + self.shortcut(x)


class CTGANGeneratorModel(nn.Module):
    def __init__(self, latent_dim: int, num_classes: int, output_dim: int, num_columns: int = None):
        """
        CTGAN生成器模型：专门为表格数据设计
        
        Args:
            latent_dim: 潜在空间维度
            num_classes: 类别数量（用于条件编码）
            output_dim: 输出特征维度
            num_columns: 列数（用于列掩码）
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.output_dim = output_dim
        self.num_columns = num_columns if num_columns else output_dim
        
        # CTGAN输入：噪声 + 条件 + 列掩码
        total_input_dim = latent_dim + num_classes + self.num_columns
        
        # CTGAN生成器架构：多个残差块
        hidden_size1 = max(256, total_input_dim)
        hidden_size2 = max(128, total_input_dim // 2)
        hidden_size3 = max(64, total_input_dim // 4)
        
        # 初始投影层
        self.input_projection = nn.Sequential(
            nn.Linear(total_input_dim, hidden_size1),
            nn.BatchNorm1d(hidden_size1),
            nn.ReLU(),
        )
        
        # 残差块
        self.residual_blocks = nn.Sequential(
            ResidualBlock(hidden_size1, hidden_size2),
            nn.ReLU(),
            ResidualBlock(hidden_size2, hidden_size3),
            nn.ReLU(),
        )
        
        # 输出层 - 使用Tanh激活（CTGAN标准做法）
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size3, output_dim),
            nn.Tanh()  # 输出范围[-1,1]，训练时会归一化到[0,1]
        )
        
        # 用于列掩码的嵌入层
        self.column_embedding = nn.Embedding(self.num_columns, self.num_columns)
        
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
    
    def _generate_column_mask(self, batch_size: int) -> torch.Tensor:
        """生成列掩码（CTGAN特性）"""
        # 随机选择一列进行掩码
        if self.training:
            col_indices = torch.randint(0, self.num_columns, (batch_size,), device=config.device)
        else:
            # 评估模式下，均匀覆盖所有列
            col_indices = torch.arange(0, batch_size, device=config.device) % self.num_columns
        
        # 转换为one-hot掩码
        mask = torch.nn.functional.one_hot(col_indices, num_classes=self.num_columns).float()
        return mask

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
        
        # 生成列掩码
        column_mask = self._generate_column_mask(num)
        
        # 拼接输入
        z_cond = torch.cat([z, condition_onehot, column_mask], dim=1)
        
        return self.forward(z_cond)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入向量 [batch_size, latent_dim + num_classes + num_columns]
            
        Returns:
            生成的样本 [batch_size, output_dim]
        """
        # 验证输入形状
        expected_dim = self.latent_dim + self.num_classes + self.num_columns
        if x.size(1) != expected_dim:
            raise ValueError(f"输入维度不匹配，期望: {expected_dim}，实际: {x.size(1)}")
        
        # 初始投影
        x = self.input_projection(x)
        
        # 残差块
        x = self.residual_blocks(x)
        
        # 输出层
        x = self.output_layer(x)
        
        return x.view(-1, self.output_dim)


class CTGANDiscriminatorModel(nn.Module):
    """
    CTGAN判别器网络：专门为表格数据设计
    """
    def __init__(self, in_features: int, num_classes: int, num_columns: int = None):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.num_columns = num_columns if num_columns else in_features
        
        # CTGAN输入：数据 + 条件 + 列掩码
        total_input_dim = in_features + num_classes + self.num_columns
        
        # CTGAN判别器架构：使用谱归一化和残差块
        hidden_size1 = max(256, total_input_dim)
        hidden_size2 = max(128, total_input_dim // 2)
        hidden_size3 = max(64, total_input_dim // 4)
        
        # 初始投影层（使用谱归一化）
        self.input_projection = nn.Sequential(
            spectral_norm(nn.Linear(total_input_dim, hidden_size1)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),  # CTGAN使用较小的dropout
        )
        
        # 残差块（使用谱归一化）
        self.residual_blocks = nn.Sequential(
            ResidualBlock(hidden_size1, hidden_size2, use_spectral_norm=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            
            ResidualBlock(hidden_size2, hidden_size3, use_spectral_norm=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
        )
        
        # 输出层
        self.output_layer = spectral_norm(nn.Linear(hidden_size3, 1))
        
        # 用于列掩码的嵌入层
        self.column_embedding = nn.Embedding(self.num_columns, self.num_columns)
        
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
        
        return torch.nn.functional.one_hot(condition, num_classes=self.num_classes).float().to(condition.device)
    
    def _generate_column_mask(self, batch_size: int) -> torch.Tensor:
        """生成列掩码（CTGAN特性）"""
        if self.training:
            col_indices = torch.randint(0, self.num_columns, (batch_size,), device=config.device)
        else:
            col_indices = torch.arange(0, batch_size, device=config.device) % self.num_columns
        
        mask = torch.nn.functional.one_hot(col_indices, num_classes=self.num_columns).float()
        return mask

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
        
        batch_size = x.size(0)
        
        # 处理条件输入
        if condition is not None:
            condition_onehot = self._process_condition(condition, target_batch_size=batch_size)
        else:
            condition_onehot = torch.zeros(batch_size, self.num_classes, device=x.device)
        
        # 生成列掩码
        column_mask = self._generate_column_mask(batch_size)
        
        # 拼接所有输入
        x_cond = torch.cat([x, condition_onehot, column_mask], dim=1)
        
        # 判别器前向传播
        x = self.input_projection(x_cond)
        x = self.residual_blocks(x)
        output = self.output_layer(x)
        
        return output
    
    def calculate_gradient_penalty(self, real_samples: torch.Tensor, fake_samples: torch.Tensor, 
                                 condition: torch.Tensor, lambda_gp: float = 10.0):
        """计算梯度惩罚（WGAN-GP）"""
        # 生成随机插值系数
        alpha = torch.rand(real_samples.size(0), 1, device=config.device)
        
        # 插值样本
        interpolated = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
        
        # 处理条件
        condition_onehot = self._process_condition(condition, target_batch_size=real_samples.size(0))
        
        # 生成列掩码
        column_mask = self._generate_column_mask(real_samples.size(0))
        
        # 拼接输入
        x_cond = torch.cat([interpolated, condition_onehot, column_mask], dim=1)
        
        # 计算判别器输出
        d_interpolated = self.output_layer(self.residual_blocks(self.input_projection(x_cond)))
        
        # 计算梯度
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # 计算梯度惩罚
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_gp
        
        return gradient_penalty


class CTGANClassifierModel(nn.Module):
    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        
        # CTGAN分类器架构
        hidden_size1 = max(256, in_features)
        hidden_size2 = max(128, in_features // 2)
        hidden_size3 = 64
        
        self.classifier_network = nn.Sequential(
            nn.Linear(in_features, hidden_size1),
            nn.BatchNorm1d(hidden_size1),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_size1, hidden_size2),
            nn.BatchNorm1d(hidden_size2),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_size2, hidden_size3),
            nn.ReLU(),
            
            # 输出层
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


class ModeSpecificNormalization(nn.Module):
    """
    模式特定归一化（CTGAN用于处理混合数据类型）
    这里简化为通用的特征归一化
    """
    def __init__(self, num_features: int, num_modes: int = 3):
        super().__init__()
        self.num_features = num_features
        self.num_modes = num_modes
        
        # 每个特征的每个模式学习不同的归一化参数
        self.gamma = nn.Parameter(torch.ones(num_modes, num_features))
        self.beta = nn.Parameter(torch.zeros(num_modes, num_features))
        
    def forward(self, x: torch.Tensor, modes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入数据 [batch_size, num_features]
            modes: 模式索引 [batch_size]
            
        Returns:
            归一化后的数据 [batch_size, num_features]
        """
        batch_size = x.size(0)
        
        # 为每个样本选择对应的模式参数
        gamma_selected = self.gamma[modes]  # [batch_size, num_features]
        beta_selected = self.beta[modes]    # [batch_size, num_features]
        
        # 应用归一化
        normalized = gamma_selected * x + beta_selected
        
        return normalized