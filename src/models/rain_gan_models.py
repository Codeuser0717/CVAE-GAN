import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm
from src import config
from src.utils import init_weights


class ResidualAttentionBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_heads: int = 4, use_spectral_norm: bool = False):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        
        self.norm1 = nn.LayerNorm(input_dim)
        self.attention = MultiHeadSelfAttention(input_dim, num_heads)
        
        self.norm2 = nn.LayerNorm(input_dim)
        
        layers = []
        if use_spectral_norm:
            layers.append(spectral_norm(nn.Linear(input_dim, output_dim)))
        else:
            layers.append(nn.Linear(input_dim, output_dim))
        layers.append(nn.LeakyReLU(0.2))
        
        if use_spectral_norm:
            layers.append(spectral_norm(nn.Linear(output_dim, output_dim)))
        else:
            layers.append(nn.Linear(output_dim, output_dim))
        
        self.feed_forward = nn.Sequential(*layers)
        
        self.shortcut = nn.Sequential()
        if input_dim != output_dim:
            if use_spectral_norm:
                self.shortcut = nn.Sequential(spectral_norm(nn.Linear(input_dim, output_dim)))
            else:
                self.shortcut = nn.Sequential(nn.Linear(input_dim, output_dim))
        
    def forward(self, x):
        x_norm = self.norm1(x)
        attn_output = self.attention(x_norm)
        x = x + attn_output
        
        x_norm = self.norm2(x)
        ff_output = self.feed_forward(x_norm)
        
        shortcut_output = self.shortcut(x)
        x = shortcut_output + ff_output
        
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 4):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_probs = torch.softmax(attn_scores, dim=-1)
        
        self.attn_probs = attn_probs
        
        attn_output = torch.matmul(attn_probs, v)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        
        return attn_output


class RAINEncoderModel(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, latent_dim: int = 128):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        
        total_input_dim = input_dim + num_classes
        
        self.input_projection = nn.Sequential(
            nn.Linear(total_input_dim, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2),
        )
        
        self.attention_blocks = nn.Sequential(
            ResidualAttentionBlock(256, 256, num_heads=4),
            nn.LeakyReLU(0.2),
            ResidualAttentionBlock(256, 128, num_heads=4),
            nn.LeakyReLU(0.2),
        )
        
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        
        self.attention_weights = None
        
        self.apply(init_weights)
    
    def _process_condition(self, condition: torch.Tensor) -> torch.Tensor:
        if condition.dim() == 1:
            condition = condition.long()
        elif condition.dim() == 2 and condition.size(1) == 1:
            condition = condition.squeeze(1).long()
        else:
            raise ValueError(f"条件输入格式错误，期望1D或2D(单列)，实际: {condition.shape}")
        
        return torch.nn.functional.one_hot(condition, num_classes=self.num_classes).float()
    
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> tuple:
        if x.dim() != 2:
            x = x.view(x.size(0), -1)
        
        condition_onehot = self._process_condition(condition)
        
        x_cond = torch.cat([x, condition_onehot], dim=1)
        
        x_cond = x_cond.unsqueeze(1)
        
        x = self.input_projection(x_cond)
        
        for block in self.attention_blocks:
            if isinstance(block, ResidualAttentionBlock):
                x = block(x)
                self.attention_weights = block.attention.attn_probs
            else:
                x = block(x)
        
        x = x.squeeze(1)
        
        mu = self.fc_mu(x)
        log_var = self.fc_logvar(x)
        
        return mu, log_var
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def encode(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        mu, log_var = self.forward(x, condition)
        return self.reparameterize(mu, log_var)


class RAINGeneratorModel(nn.Module):
    def __init__(self, latent_dim: int, num_classes: int, output_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.output_dim = output_dim
        
        total_input_dim = latent_dim + num_classes
        
        self.input_projection = nn.Sequential(
            nn.Linear(total_input_dim, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2),
        )
        
        self.attention_blocks = nn.Sequential(
            ResidualAttentionBlock(256, 256, num_heads=4),
            nn.LeakyReLU(0.2),
            ResidualAttentionBlock(256, 128, num_heads=4),
            nn.LeakyReLU(0.2),
            ResidualAttentionBlock(128, 64, num_heads=4),
            nn.LeakyReLU(0.2),
        )
        
        self.output_layer = nn.Sequential(
            nn.Linear(64, output_dim),
            nn.Sigmoid()
        )
        
        self.attention_weights = None
        
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
        
        z = torch.randn(num, self.latent_dim, device=config.device)
        
        z_cond = torch.cat([z, condition_onehot], dim=1)
        
        z_cond = z_cond.unsqueeze(1)
        
        return self.forward(z_cond).squeeze(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        
        for block in self.attention_blocks:
            if isinstance(block, ResidualAttentionBlock):
                x = block(x)
                self.attention_weights = block.attention.attn_probs
            else:
                x = block(x)
        
        x = self.output_layer(x)
        
        return x


class RAINDiscriminatorModel(nn.Module):
    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        
        total_input_dim = in_features + num_classes
        
        self.input_projection = nn.Sequential(
            spectral_norm(nn.Linear(total_input_dim, 256)),
            nn.LeakyReLU(0.2),
        )
        
        self.attention_blocks = nn.Sequential(
            ResidualAttentionBlock(256, 256, num_heads=4, use_spectral_norm=True),
            nn.LeakyReLU(0.2),
            ResidualAttentionBlock(256, 128, num_heads=4, use_spectral_norm=True),
            nn.LeakyReLU(0.2),
        )
        
        self.output_layer = spectral_norm(nn.Linear(128, 1))
        
        self.attention_weights = None
        
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
        
        batch_size = x.size(0)
        
        if condition is not None:
            condition_onehot = self._process_condition(condition, target_batch_size=batch_size)
        else:
            condition_onehot = torch.zeros(batch_size, self.num_classes, device=x.device)
        
        x_cond = torch.cat([x, condition_onehot], dim=1)
        
        x_cond = x_cond.unsqueeze(1)
        
        x = self.input_projection(x_cond)
        
        for block in self.attention_blocks:
            if isinstance(block, ResidualAttentionBlock):
                x = block(x)
                self.attention_weights = block.attention.attn_probs
            else:
                x = block(x)
        
        x = x.squeeze(1)
        
        output = self.output_layer(x)
        
        return output


class RAINClassifierModel(nn.Module):
    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        
        self.input_projection = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
        )
        
        self.attention_blocks = nn.Sequential(
            ResidualAttentionBlock(256, 256, num_heads=4),
            nn.ReLU(),
            ResidualAttentionBlock(256, 128, num_heads=4),
            nn.ReLU(),
        )
        
        self.output_layer = nn.Sequential(
            nn.Linear(128, num_classes),
        )
        
        self.attention_weights = None
        
        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        x = x.unsqueeze(1)
        
        x = self.input_projection(x)
        
        for block in self.attention_blocks:
            if isinstance(block, ResidualAttentionBlock):
                x = block(x)
                self.attention_weights = block.attention.attn_probs
            else:
                x = block(x)
        
        x = x.squeeze(1)
        
        x = self.output_layer(x)
        
        return x
    
    def get_attention_weights(self):
        return self.attention_weights