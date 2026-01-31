import torch
from torch import nn
from src import config
from src.utils import init_weights


class VAEEncoderModel(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 128):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
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
        
        self.fc_mu = nn.Linear(hidden_size3, latent_dim)
        self.fc_logvar = nn.Linear(hidden_size3, latent_dim)
        
        self.apply(init_weights)
    
    def forward(self, x: torch.Tensor) -> tuple:
        if x.dim() != 2:
            raise ValueError(f"输入数据应为2D张量，实际: {x.shape}")
        if x.size(1) != self.input_dim:
            raise ValueError(f"输入特征维度不匹配，期望: {self.input_dim}，实际: {x.size(1)}")
        
        hidden = self.encoder(x)
        
        mu = self.fc_mu(hidden)
        log_var = self.fc_logvar(hidden)
        
        return mu, log_var
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        mu, log_var = self.forward(x)
        return self.reparameterize(mu, log_var)


class VAEDecoderModel(nn.Module):
    def __init__(self, latent_dim: int, output_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        
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
        
        self.last_layer = nn.Sequential(
            nn.Linear(hidden_size3, output_dim),
            nn.Sigmoid()
        )
        
        self.apply(init_weights)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() != 2:
            raise ValueError(f"潜在向量应为2D张量，实际: {z.shape}")
        if z.size(1) != self.latent_dim:
            raise ValueError(f"潜在维度不匹配，期望: {self.latent_dim}，实际: {z.size(1)}")
        
        x = self.main_model(z)
        self.hidden_status = x
        x = self.last_layer(x)
        
        return x.view(-1, self.output_dim)
    
    def reconstruct(self, x: torch.Tensor, encoder: nn.Module) -> torch.Tensor:
        with torch.no_grad():
            z = encoder.encode(x)
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
            nn.LayerNorm(hidden_size2),
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