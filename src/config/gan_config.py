epochs: int = 500
batch_size: int = 128

z_size: int = 128

g_lr: float = 2e-4
g_loop_num: int = 3

d_lr: float = 2e-4
d_loop_num: int = 5

c_lr: float = 1e-4
c_loop_num: int = 5

cvae_gan_config = {
    'lambda_recon': 1.0,      # 重构损失权重
    'lambda_kl': 0.1,        # KL散度损失权重  
    'lambda_adv': 1.0,        # 对抗损失权重
    'lambda_class': 0.5,      # 分类损失权重
    'confidence_threshold': 0.5,  # 合格样本置信度阈值
}
ae_gan_config = {
    'lambda_recon': 1.0,
    'lambda_adv': 1.0,
}
cae_gan_config = {
    'lambda_recon': 1.0,
    'lambda_adv': 1.0,
    'lambda_class': 0.5,
    'confidence_threshold': 0.5,
}

vae_gan_config = {
    'lambda_recon': 1.0,    
    'lambda_kl': 0.01,      
    'lambda_adv': 0.1,     
    'confidence_threshold': 0.5,  
}

cgan_config = {
    'lambda_adv': 1.0,      
    'lambda_class': 0.5,    
    'confidence_threshold': 0.5, 
}

gan_config = {
    'lambda_adv': 1.0,      
    'confidence_threshold': 0.5,  
}

cvae_config = {
    'lambda_recon': 1.0,    
    'lambda_kl': 0.01,      
    'lambda_class': 0.1,    
    'confidence_threshold': 0.5,  
}

vae_config = {
    'lambda_recon': 1.0,   
    'lambda_kl': 0.01,    
    'confidence_threshold': 0.5,  
}

sngan_config = {
    'lambda_adv': 1.0,      
    'lambda_class': 0.5,   
    'confidence_threshold': 0.5,  
}

qg_smote_config = {
    'num_quantiles': 3,         
    'lambda_recon': 1.0,        
    'lambda_quantile': 0.5,      
    'lambda_adv': 0.1,           
    'lambda_class': 0.1,         
    'confidence_threshold': 0.5, 
}

ctgan_config = {
    'lambda_adv': 1.0,      
    'lambda_class': 0.5,    
    'lambda_gp': 10.0,      
    'confidence_threshold': 0.5,  
}

rain_gan_config = {
    'lambda_recon': 1.0,        
    'lambda_kl': 0.01,          
    'lambda_adv': 0.1,          
    'lambda_class': 0.1,        
    'lambda_attention': 0.01,    
    'confidence_threshold': 0.5,  
}