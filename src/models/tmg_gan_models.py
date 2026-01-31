import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm

from src import config
from src.utils import init_weights


class TMGGANCDModel(nn.Module):
    def __init__(self, in_features: int, label_num: int):
        super().__init__()
        
        hidden_size1 = max(256, in_features)
        hidden_size2 = max(128, in_features // 2)
        hidden_size3 = 64
        
        self.main_model = nn.Sequential(
            spectral_norm(nn.Linear(in_features, hidden_size1)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            spectral_norm(nn.Linear(hidden_size1, hidden_size2)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            spectral_norm(nn.Linear(hidden_size2, hidden_size3)),
            nn.LeakyReLU(0.2),
        )
        
        self.hidden_status: torch.Tensor = None
        self.c_last_layer = nn.Sequential(
            nn.Linear(hidden_size3, label_num),
            nn.Softmax(dim=1),
        )
        self.d_last_layer = nn.Sequential(
            spectral_norm(nn.Linear(hidden_size3, 1)),
        )
        
        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
            
        x = self.main_model(x)
        self.hidden_status = x
        return self.d_last_layer(x), self.c_last_layer(x)

class TMGGANGeneratorModel(nn.Module):
    def __init__(self, z_size: int, feature_num: int):
        super().__init__()
        self.z_size = z_size
        self.feature_num = feature_num
        
        hidden_size1 = max(256, feature_num)
        hidden_size2 = max(128, feature_num // 2)
        hidden_size3 = max(64, feature_num // 4)
        
        self.main_model = nn.Sequential(
            nn.Linear(z_size, hidden_size1),
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
            nn.Linear(hidden_size3, feature_num),
            nn.Tanh()
        )
        
        self.apply(init_weights)

    def generate_samples(self, num: int) -> torch.Tensor:
        z = torch.randn(num, self.z_size, device=config.device)
        return self.forward(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        x = self.main_model(x)
        self.hidden_status = x
        x = self.last_layer(x)
        
        return x.view(-1, self.feature_num)

