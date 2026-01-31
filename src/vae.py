import pickle
import torch
import src
from src import Classifier, datasets, utils, models, config
import time
import numpy as np
import os


class VAE:
    def __init__(self):
        self.feature_num = datasets.feature_num
        
        self.encoder = models.VAEEncoderModel(
            input_dim=self.feature_num, 
            latent_dim=config.gan_config.z_size
        ).to(config.device)
        
        self.decoder = models.VAEDecoderModel(
            latent_dim=config.gan_config.z_size,
            output_dim=self.feature_num
        ).to(config.device)
        
        self.classifier = models.ClassifierModel(
            in_features=self.feature_num,
            num_classes=datasets.label_num
        ).to(config.device)
        

        self.samples = None
        self.labels = None
        
        # 从配置文件加载损失权重
        self.lambda_recon = config.gan_config.vae_config['lambda_recon']
        self.lambda_kl = config.gan_config.vae_config['lambda_kl']
        
        # 用于记录训练过程中的损失
        self.loss_history = {
            'recon_loss': [],
            'kl_loss': []
        }

    def fit(self, dataset):
        """训练VAE模型
        
        Args:
            dataset: 训练数据集
        """
        # 设置所有模型为训练模式
        self.encoder.train()
        self.decoder.train()
        self.classifier.train()

        # 存储所有训练样本和标签
        self._store_samples(dataset)

        # 创建优化器
        encoder_optimizer = torch.optim.Adam(
            params=self.encoder.parameters(),
            lr=config.gan_config.g_lr,
            betas=(0.5, 0.999),
        )
        
        decoder_optimizer = torch.optim.Adam(
            params=self.decoder.parameters(),
            lr=config.gan_config.g_lr,
            betas=(0.5, 0.999),
        )

        classifier_optimizer = torch.optim.Adam(
            params=self.classifier.parameters(),
            lr=config.gan_config.c_lr,
            betas=(0.5, 0.999),
        )

        # 训练循环
        for e in range(config.gan_config.epochs):
            # 训练分类器多次（如果需要）
            if config.gan_config.c_loop_num > 0:
                for _ in range(config.gan_config.c_loop_num):
                    classifier_optimizer.zero_grad()
                    
                    # 获取真实样本和标签
                    real_samples, real_labels = self._get_random_samples_with_labels(config.gan_config.batch_size)
                    
                    # 计算分类损失
                    classifier_output = self.classifier(real_samples)
                    c_loss = torch.nn.functional.cross_entropy(classifier_output, real_labels)
                    
                    c_loss.backward()
                    classifier_optimizer.step()
                
            # 训练编码器和解码器多次
            for _ in range(config.gan_config.g_loop_num):
                encoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()
                
                # 获取真实样本
                real_samples = self._get_random_samples(config.gan_config.batch_size)
                
                # 编码器前向传播
                mu, log_var = self.encoder(real_samples)
                z = self.encoder.reparameterize(mu, log_var)
                
                # 解码器重构样本
                x_recon = self.decoder(z)
                
                # 计算损失
                recon_loss = torch.nn.functional.mse_loss(x_recon, real_samples)
                kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / mu.size(0)
                
                # 总损失
                total_loss = self.lambda_recon * recon_loss + self.lambda_kl * kl_loss
                
                total_loss.backward()
                encoder_optimizer.step()
                decoder_optimizer.step()

            # 记录损失值
            self.loss_history['recon_loss'].append(recon_loss.item())
            self.loss_history['kl_loss'].append(kl_loss.item())
            
            # 每50轮打印一次训练进度
            if e % 50 == 0:
                print(f"VAE训练轮次: {e}/{config.gan_config.epochs}, "
                      f"重构损失: {recon_loss.item():.4f}, "
                      f"KL损失: {kl_loss.item():.4f}")
            
        # 设置所有模型为评估模式
        self.encoder.eval()
        self.decoder.eval()
        self.classifier.eval()

    def _store_samples(self, dataset):
        """存储所有训练样本和标签"""
        samples_list = []
        labels_list = []
        for sample, label in dataset:
            samples_list.append(sample.unsqueeze(0))
            labels_list.append(label.unsqueeze(0))
        self.samples = torch.cat(samples_list, dim=0)
        self.labels = torch.cat(labels_list, dim=0)

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
    
    def _get_random_samples_with_labels(self, num: int) -> tuple:
        """随机选择指定数量的样本和对应标签"""
        if len(self.samples) < num:
            # 如果可用样本数量不足，进行有放回采样
            indices = torch.randint(0, len(self.samples), (num,))
        elif len(self.samples) == num:
            # 如果可用样本数量刚好等于请求数量，返回所有样本
            indices = torch.arange(len(self.samples))
        else:
            # 如果可用样本数量大于请求数量，进行无放回采样
            indices = torch.randperm(len(self.samples))[:num]
        
        return self.samples[indices], self.labels[indices]

    def plot_loss_history(self):
        """绘制损失历史曲线并保存为图片"""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 6))
        
        # 绘制重构损失
        plt.subplot(1, 2, 1)
        plt.plot(self.loss_history['recon_loss'], color='blue')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Reconstruction Loss')
        
        # 绘制KL损失
        plt.subplot(1, 2, 2)
        plt.plot(self.loss_history['kl_loss'], color='green')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('KL divergence loss')
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图片
        loss_plot_path = config.path_config.gan_outs / 'vae_loss_history.jpg'
        plt.savefig(loss_plot_path)
        plt.close()
        
        print(f"\n损失曲线已保存至: {loss_plot_path}")
        
        # 同时绘制一个综合图
        plt.figure(figsize=(12, 6))
        
        plt.plot(self.loss_history['recon_loss'], label='重构损失', color='blue')
        plt.plot(self.loss_history['kl_loss'], label='KL散度', color='green')
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('VAE损失曲线')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 保存综合图片
        combined_loss_path = config.path_config.gan_outs / 'vae_combined_loss.jpg'
        plt.savefig(combined_loss_path)
        plt.close()
        
        print(f"综合损失曲线已保存至: {combined_loss_path}")

    def generate_samples(self, num: int):
        """生成指定数量的样本 - 使用先验分布"""
        z_prior = torch.randn(num, config.gan_config.z_size, device=config.device)
        return self.decoder(z_prior).cpu().detach()

    def generate_qualified_samples(self, target_label: int, num: int, confidence_threshold: float = None):
        """生成经过分类器验证的合格样本（需要指定目标标签）"""
        result = []
        patience = 20
        # 如果没有提供置信度阈值，则使用配置文件中的值
        if confidence_threshold is None:
            confidence_threshold = config.gan_config.vae_config['confidence_threshold']
        
        while len(result) < num and patience > 0:
            # 生成样本
            samples = self.generate_samples(min(10, num - len(result)))
            
            # 使用分类器验证
            with torch.no_grad():
                self.classifier.eval()
                classifier_output = self.classifier(samples.to(config.device))
                self.classifier.train()
                
                # 获取预测概率
                probs = torch.softmax(classifier_output, dim=1)
                max_probs, preds = torch.max(probs, dim=1)
                
                # 筛选高置信度且类别正确的样本
                valid_mask = (max_probs > confidence_threshold) & (preds == target_label)
                valid_samples = samples[valid_mask.cpu()]
                
                result.extend(valid_samples)
                
                if len(valid_samples) == 0:
                    patience -= 1
        
        return torch.stack(result) if result else torch.tensor([])

    def reconstruct_samples(self, samples: torch.Tensor):
        """重构输入样本 - 使用编码器得到的潜在向量"""
        with torch.no_grad():
            self.encoder.eval()
            self.decoder.eval()
            
            samples = samples.to(config.device)
            
            # 使用编码器得到潜在向量
            z = self.encoder.encode(samples)
            # 使用解码器重构
            reconstructed = self.decoder(z)
            
            self.encoder.train()
            self.decoder.train()
            
            return reconstructed.cpu().detach()