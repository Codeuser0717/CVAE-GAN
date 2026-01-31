import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm
from src import config
from src.utils import init_weights


class SNGANGeneratorModel(nn.Module):
    def __init__(self, latent_dim: int, num_classes: int, output_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.output_dim = output_dim
        
        total_input_dim = latent_dim + num_classes
        
        hidden_size1 = max(256, total_input_dim)
        hidden_size2 = max(128, total_input_dim // 2)
        hidden_size3 = max(64, total_input_dim // 4)
        
        self.main_model = nn.Sequential(
            spectral_norm(nn.Linear(total_input_dim, hidden_size1)),
            nn.BatchNorm1d(hidden_size1),
            nn.ReLU(),
            
            spectral_norm(nn.Linear(hidden_size1, hidden_size2)),
            nn.BatchNorm1d(hidden_size2),
            nn.ReLU(),
            
            spectral_norm(nn.Linear(hidden_size2, hidden_size3)),
            nn.BatchNorm1d(hidden_size3),
            nn.ReLU(),
        )
        
        self.hidden_status: torch.Tensor = None
        
        self.last_layer = nn.Sequential(
            spectral_norm(nn.Linear(hidden_size3, output_dim)),
            nn.Sigmoid()
        )
        
        self.apply(init_weights)
    
    def _process_condition(self, condition: torch.Tensor, target_batch_size: int = None) -> torch.Tensor:
        if condition.dim() == 0:
            condition = condition.unsqueeze(0)
        
        if condition.dim() == 1:
            condition = condition.long()
            if target_batch_size and condition.size(0) == 1:
                condition = condition.repeat(target_batch_size)
        elif condition.dim() == 2 and condition.size(1) == 1:
            condition = condition.squeeze(1).long()
        else:
            raise ValueError(f"条件输入格式错误: {condition.shape}")
        
        return torch.nn.functional.one_hot(condition, num_classes=self.num_classes).float()

    def generate_conditional_samples(self, num: int, condition: torch.Tensor) -> torch.Tensor:
        condition_onehot = self._process_condition(condition, target_batch_size=num)
        
        if condition_onehot.size(0) != num:
            raise ValueError(f"条件数量不匹配，期望: {num}，实际: {condition_onehot.size(0)}")
        
        z = torch.randn(num, self.latent_dim, device=config.device)
        return self.forward(z, condition_onehot)

    def forward(self, z: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
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
        
        z_cond = torch.cat([z, condition], dim=1)
        
        x = self.main_model(z_cond)
        self.hidden_status = x
        x = self.last_layer(x)
        
        return x.view(-1, self.output_dim)


class SNGANDiscriminatorModel(nn.Module):
    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        
        total_input_dim = in_features + num_classes
        
        hidden_size1 = max(256, total_input_dim)
        hidden_size2 = max(128, total_input_dim // 2)
        hidden_size3 = 64
        
        self.discriminator_network = nn.Sequential(
            spectral_norm(nn.Linear(total_input_dim, hidden_size1)),
            nn.LeakyReLU(0.2),
            
            spectral_norm(nn.Linear(hidden_size1, hidden_size2)),
            nn.LeakyReLU(0.2),
            
            spectral_norm(nn.Linear(hidden_size2, hidden_size3)),
            nn.LeakyReLU(0.2),
            
            spectral_norm(nn.Linear(hidden_size3, 1)),
        )
        
        self.hidden_status: torch.Tensor = None
        self.apply(init_weights)
    
    def _process_condition(self, condition: torch.Tensor, target_batch_size: int = None) -> torch.Tensor:
        if condition.dim() == 0:
            condition = condition.unsqueeze(0)
        
        if condition.dim() == 1:
            condition = condition.long()
            if target_batch_size:
                if condition.size(0) == 1:
                    condition = condition.repeat(target_batch_size)
                elif condition.size(0) != target_batch_size:
                    raise ValueError(f"条件batch大小不匹配，期望: {target_batch_size}，实际: {condition.size(0)}")
        elif condition.dim() == 2 and condition.size(1) == 1:
            condition = condition.squeeze(1).long()
            if target_batch_size and condition.size(0) != target_batch_size:
                raise ValueError(f"条件batch大小不匹配，期望: {target_batch_size}，实际: {condition.size(0)}")
        else:
            raise ValueError(f"条件输入格式错误: {condition.shape}")
        
        return torch.nn.functional.one_hot(condition, num_classes=self.num_classes).float()

    def forward(self, x: torch.Tensor, condition: torch.Tensor = None) -> torch.Tensor:
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        if condition is not None:
            condition_onehot = self._process_condition(condition, target_batch_size=x.size(0))
            x_cond = torch.cat([x, condition_onehot], dim=1)
        else:
            condition_onehot = torch.zeros(x.size(0), self.num_classes, device=x.device)
            x_cond = torch.cat([x, condition_onehot], dim=1)
        
        features = self.discriminator_network[:-1](x_cond)
        self.hidden_status = features
        output = self.discriminator_network[-1](features)
        
        return output


class SNGANClassifierModel(nn.Module):
    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        
        hidden_size1 = max(256, in_features)
        hidden_size2 = max(128, in_features // 2)
        hidden_size3 = 64
        
        self.classifier_network = nn.Sequential(
            spectral_norm(nn.Linear(in_features, hidden_size1)),
            nn.ReLU(),
            
            spectral_norm(nn.Linear(hidden_size1, hidden_size2)),
            nn.ReLU(),
            
            spectral_norm(nn.Linear(hidden_size2, hidden_size3)),
            nn.ReLU(),
            
            spectral_norm(nn.Linear(hidden_size3, num_classes)),
        )
        
        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.classifier_network(x)
    
    def get_feature_importance(self, x: torch.Tensor):
        with torch.no_grad():
            first_layer = self.classifier_network[0]
            if hasattr(first_layer, 'weight'):
                weights = first_layer.weight.data
                feature_importance = torch.mean(torch.abs(weights), dim=0)
                return feature_importance
        return None