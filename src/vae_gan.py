import pickle
import torch
import src
from src import Classifier, datasets, utils, models, config
import time
import numpy as np
import os


class VAEGAN:
    def __init__(self):
        self.feature_num = datasets.feature_num
        
        self.encoder = models.VAEGANEncoderModel(
            input_dim=self.feature_num, 
            latent_dim=config.gan_config.z_size
        ).to(config.device)
        
        self.generator = models.VAEGANGeneratorModel(
            latent_dim=config.gan_config.z_size,
            output_dim=self.feature_num
        ).to(config.device)
        
        self.discriminator = models.VAEGANDiscriminatorModel(
            in_features=self.feature_num
        ).to(config.device)
        
        self.samples = None
        
        # 从配置文件加载损失权重
        self.lambda_recon = config.gan_config.vae_gan_config['lambda_recon']
        self.lambda_kl = config.gan_config.vae_gan_config['lambda_kl']
        self.lambda_adv = config.gan_config.vae_gan_config['lambda_adv']
        
        # 用于记录训练过程中的损失
        self.loss_history = {
            'recon_loss': [],
            'kl_loss': [],
            'adv_loss': []
        }

    def fit(self, dataset):
        """训练VAE-GAN模型
        
        Args:
            dataset: 训练数据集
        """
        # 设置所有模型为训练模式
        self.encoder.train()
        self.generator.train()
        self.discriminator.train()

        # 存储所有训练样本（不按标签分类）
        self._store_samples(dataset)

        # 创建优化器
        encoder_optimizer = torch.optim.Adam(
            params=self.encoder.parameters(),
            lr=config.gan_config.g_lr,
            betas=(0.5, 0.999),
        )
        
        generator_optimizer = torch.optim.Adam(
            params=self.generator.parameters(),
            lr=config.gan_config.g_lr,
            betas=(0.5, 0.999),
        )
        
        discriminator_optimizer = torch.optim.Adam(
            params=self.discriminator.parameters(),
            lr=config.gan_config.d_lr,
            betas=(0.5, 0.999),
        )

        # 训练循环
        for e in range(config.gan_config.epochs):
            # 训练判别器多次
            for _ in range(config.gan_config.d_loop_num):
                discriminator_optimizer.zero_grad()
                
                # 获取真实样本
                real_samples = self._get_random_samples(config.gan_config.batch_size)
                
                # 从先验分布采样生成样本
                with torch.no_grad():
                    z_prior = torch.randn(config.gan_config.batch_size, config.gan_config.z_size, device=config.device)
                    generated_samples = self.generator(z_prior)
                
                # 判别器对真实样本的输出
                d_real = self.discriminator(real_samples)
                d_real_loss = -d_real.mean()
                
                # 判别器对生成样本的输出
                d_fake = self.discriminator(generated_samples.detach())
                d_fake_loss = d_fake.mean()

                # 判别器总损失
                d_loss = d_real_loss + d_fake_loss
                d_loss.backward()
                discriminator_optimizer.step()
                
            # 训练编码器和生成器多次
            for _ in range(config.gan_config.g_loop_num):
                encoder_optimizer.zero_grad()
                generator_optimizer.zero_grad()
                
                # 获取真实样本
                real_samples = self._get_random_samples(config.gan_config.batch_size)
                
                # 1. 从编码器得到 z_enc（用于重构和KL损失）
                mu, log_var = self.encoder(real_samples)
                z_enc = self.encoder.reparameterize(mu, log_var)
                
                # 2. 从先验分布采样 z_prior（用于对抗损失）
                z_prior = torch.randn(config.gan_config.batch_size, config.gan_config.z_size, device=config.device)
                
                # 3. 使用 z_enc 计算重构样本（用于重构和KL损失）
                x_recon = self.generator(z_enc)
                
                # 4. 使用 z_prior 计算生成样本（用于对抗损失）
                x_fake = self.generator(z_prior)
                
                # 计算各项损失
                # 重构损失和KL损失使用编码器得到的 z_enc
                recon_loss = torch.nn.functional.mse_loss(x_recon, real_samples)
                kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / mu.size(0)
                
                # 对抗损失使用判别器
                d_fake = self.discriminator(x_fake)
                adv_loss = -d_fake.mean()
                
                # 总损失
                total_loss = (
                    config.gan_config.vae_gan_config['lambda_recon'] * recon_loss +
                    config.gan_config.vae_gan_config['lambda_kl'] * kl_loss +
                    config.gan_config.vae_gan_config['lambda_adv'] * adv_loss
                )
                
                total_loss.backward()
                encoder_optimizer.step()
                generator_optimizer.step()

            # 记录损失值
            self.loss_history['recon_loss'].append(recon_loss.item())
            self.loss_history['kl_loss'].append(kl_loss.item())
            self.loss_history['adv_loss'].append(adv_loss.item())
            
            # 每50轮打印一次训练进度
            if e % 50 == 0:
                print(f"VAE-GAN训练轮次: {e}/{config.gan_config.epochs}, "
                      f"重构损失: {recon_loss.item():.4f}, "
                      f"KL损失: {kl_loss.item():.4f}, "
                      f"对抗损失: {adv_loss.item():.4f}")
            
        # 设置所有模型为评估模式
        self.encoder.eval()
        self.generator.eval()
        self.discriminator.eval()

    def _store_samples(self, dataset):
        """存储所有训练样本（不分标签）"""
        samples_list = []
        for sample, _ in dataset:  # 忽略标签
            samples_list.append(sample.unsqueeze(0))
        self.samples = torch.cat(samples_list, dim=0)

    def _get_random_samples(self, num: int) -> torch.Tensor:
        """随机选择指定数量的样本"""
        if len(self.samples) < num:
            # 如果可用样本数量不足，进行有放回采样
            indices = torch.randint(0, len(self.samples), (num,))
            return self.samples[indices]
        elif len(self.samples) == num:
            # 如果可用样本数量刚好等于请求数量，返回所有样本
            return self.samples
        else:
            # 如果可用样本数量大于请求数量，进行无放回采样
            indices = torch.randperm(len(self.samples))[:num]
            return self.samples[indices]

    def plot_loss_history(self):
        """绘制损失历史曲线并保存为图片"""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 8))
        
        # 绘制重构损失
        plt.subplot(2, 2, 1)
        plt.plot(self.loss_history['recon_loss'], color='blue')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Reconstruction Loss')
        
        # 绘制KL损失
        plt.subplot(2, 2, 2)
        plt.plot(self.loss_history['kl_loss'], color='green')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('KL divergence loss')
        
        # 绘制对抗损失
        plt.subplot(2, 2, 3)
        plt.plot(self.loss_history['adv_loss'], color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Adversarial Loss')
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图片
        loss_plot_path = config.path_config.gan_outs / 'vae_gan_loss_history.jpg'
        plt.savefig(loss_plot_path)
        plt.close()
        
        print(f"\n损失曲线已保存至: {loss_plot_path}")
        
        # 同时绘制一个综合图
        plt.figure(figsize=(12, 6))
        
        # 为了在同一图上显示，对对抗损失取绝对值
        adv_loss_abs = [abs(loss) for loss in self.loss_history['adv_loss']]
        
        plt.plot(self.loss_history['recon_loss'], label='重构损失', color='blue')
        plt.plot(self.loss_history['kl_loss'], label='KL散度', color='green')
        plt.plot(adv_loss_abs, label='对抗损失(绝对值)', color='red')
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('VAE-GAN损失曲线')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 保存综合图片
        combined_loss_path = config.path_config.gan_outs / 'vae_gan_combined_loss.jpg'
        plt.savefig(combined_loss_path)
        plt.close()
        
        print(f"综合损失曲线已保存至: {combined_loss_path}")

    def generate_samples(self, num: int):
        """生成指定数量的样本 - 使用先验分布"""
        z_prior = torch.randn(num, config.gan_config.z_size, device=config.device)
        return self.generator(z_prior).cpu().detach()

    def reconstruct_samples(self, samples: torch.Tensor):
        """重构输入样本 - 使用编码器得到的潜在向量"""
        with torch.no_grad():
            self.encoder.eval()
            self.generator.eval()
            
            samples = samples.to(config.device)
            
            # 使用编码器得到潜在向量
            z_enc = self.encoder.encode(samples)
            # 使用生成器重构
            reconstructed = self.generator(z_enc)
            
            self.encoder.train()
            self.generator.train()
            
            return reconstructed.cpu().detach()