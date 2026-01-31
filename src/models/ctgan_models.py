import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm
from src import config
from src.utils import init_weights


class ResidualBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, use_spectral_norm: bool = False):
        super().__init__()
        
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
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.output_dim = output_dim
        self.num_columns = num_columns if num_columns else output_dim
        
        total_input_dim = latent_dim + num_classes + self.num_columns
        
        hidden_size1 = max(256, total_input_dim)
        hidden_size2 = max(128, total_input_dim // 2)
        hidden_size3 = max(64, total_input_dim // 4)
        
        self.input_projection = nn.Sequential(
            nn.Linear(total_input_dim, hidden_size1),
            nn.BatchNorm1d(hidden_size1),
            nn.ReLU(),
        )
        
        self.residual_blocks = nn.Sequential(
            ResidualBlock(hidden_size1, hidden_size2),
            nn.ReLU(),
            ResidualBlock(hidden_size2, hidden_size3),
            nn.ReLU(),
        )
        
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size3, output_dim),
            nn.Tanh()
        )
        
        self.column_embedding = nn.Embedding(self.num_columns, self.num_columns)
        
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
    
    def _generate_column_mask(self, batch_size: int) -> torch.Tensor:
        if self.training:
            col_indices = torch.randint(0, self.num_columns, (batch_size,), device=config.device)
        else:
            col_indices = torch.arange(0, batch_size, device=config.device) % self.num_columns
        
        mask = torch.nn.functional.one_hot(col_indices, num_classes=self.num_columns).float()
        return mask

    def generate_conditional_samples(self, num: int, condition: torch.Tensor) -> torch.Tensor:
        condition_onehot = self._process_condition(condition, target_batch_size=num)
        
        if condition_onehot.size(0) != num:
            raise ValueError(f"条件数量不匹配，期望: {num}，实际: {condition_onehot.size(0)}")
        
        z = torch.randn(num, self.latent_dim, device=config.device)
        
        column_mask = self._generate_column_mask(num)
        
        z_cond = torch.cat([z, condition_onehot, column_mask], dim=1)
        
        return self.forward(z_cond)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        expected_dim = self.latent_dim + self.num_classes + self.num_columns
        if x.size(1) != expected_dim:
            raise ValueError(f"输入维度不匹配，期望: {expected_dim}，实际: {x.size(1)}")
        
        x = self.input_projection(x)
        
        x = self.residual_blocks(x)
        
        x = self.output_layer(x)
        
        return x.view(-1, self.output_dim)


class CTGANDiscriminatorModel(nn.Module):
    def __init__(self, in_features: int, num_classes: int, num_columns: int = None):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.num_columns = num_columns if num_columns else in_features
        
        total_input_dim = in_features + num_classes + self.num_columns
        
        hidden_size1 = max(256, total_input_dim)
        hidden_size2 = max(128, total_input_dim // 2)
        hidden_size3 = max(64, total_input_dim // 4)
        
        self.input_projection = nn.Sequential(
            spectral_norm(nn.Linear(total_input_dim, hidden_size1)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
        )
        
        self.residual_blocks = nn.Sequential(
            ResidualBlock(hidden_size1, hidden_size2, use_spectral_norm=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            
            ResidualBlock(hidden_size2, hidden_size3, use_spectral_norm=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
        )
        
        self.output_layer = spectral_norm(nn.Linear(hidden_size3, 1))
        
        self.column_embedding = nn.Embedding(self.num_columns, self.num_columns)
        
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
        
        return torch.nn.functional.one_hot(condition, num_classes=self.num_classes).float().to(condition.device)
    
    def _generate_column_mask(self, batch_size: int) -> torch.Tensor:
        if self.training:
            col_indices = torch.randint(0, self.num_columns, (batch_size,), device=config.device)
        else:
            col_indices = torch.arange(0, batch_size, device=config.device) % self.num_columns
        
        mask = torch.nn.functional.one_hot(col_indices, num_classes=self.num_columns).float()
        return mask

    def forward(self, x: torch.Tensor, condition: torch.Tensor = None) -> torch.Tensor:
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        batch_size = x.size(0)
        
        if condition is not None:
            condition_onehot = self._process_condition(condition, target_batch_size=batch_size)
        else:
            condition_onehot = torch.zeros(batch_size, self.num_classes, device=x.device)
        
        column_mask = self._generate_column_mask(batch_size)
        
        x_cond = torch.cat([x, condition_onehot, column_mask], dim=1)
        
        x = self.input_projection(x_cond)
        x = self.residual_blocks(x)
        output = self.output_layer(x)
        
        return output
    
    def calculate_gradient_penalty(self, real_samples: torch.Tensor, fake_samples: torch.Tensor, 
                                 condition: torch.Tensor, lambda_gp: float = 10.0):
        alpha = torch.rand(real_samples.size(0), 1, device=config.device)
        
        interpolated = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
        
        condition_onehot = self._process_condition(condition, target_batch_size=real_samples.size(0))
        
        column_mask = self._generate_column_mask(real_samples.size(0))
        
        x_cond = torch.cat([interpolated, condition_onehot, column_mask], dim=1)
        
        d_interpolated = self.output_layer(self.residual_blocks(self.input_projection(x_cond)))
        
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_gp
        
        return gradient_penalty


class CTGANClassifierModel(nn.Module):
    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        
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
            
            nn.Linear(hidden_size3, num_classes),
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


class ModeSpecificNormalization(nn.Module):
    def __init__(self, num_features: int, num_modes: int = 3):
        super().__init__()
        self.num_features = num_features
        self.num_modes = num_modes
        
        self.gamma = nn.Parameter(torch.ones(num_modes, num_features))
        self.beta = nn.Parameter(torch.zeros(num_modes, num_features))
        
    def forward(self, x: torch.Tensor, modes: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        gamma_selected = self.gamma[modes]
        beta_selected = self.beta[modes]
        
        normalized = gamma_selected * x + beta_selected
        
        return normalized