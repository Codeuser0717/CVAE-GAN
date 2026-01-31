import context
import pickle
import torch
import src
from src import Classifier, datasets, utils, models, config
import time
import numpy as np
import os


class CVAE:
    
    def __init__(self):
        """初始化CVAE模型"""
        # 获取数据特征和类别信息
        self.feature_num = datasets.feature_num
        self.label_num = datasets.label_num
        
        self.encoder = models.CVAEEncoderModel(
            input_dim=self.feature_num, 
            num_classes=self.label_num,
            latent_dim=config.gan_config.z_size
        ).to(config.device)
        
        self.generator = models.CVAEGeneratorModel(
            latent_dim=config.gan_config.z_size,
            num_classes=self.label_num,
            output_dim=self.feature_num
        ).to(config.device)
        
        self.classifier = models.CVAEClassifierModel(
            in_features=self.feature_num,
            num_classes=self.label_num
        ).to(config.device)
        
        # 用于存储按标签分类的样本字典
        self.samples = dict()
        
        # 从配置文件加载损失权重
        self.lambda_recon = config.gan_config.cvae_config['lambda_recon']
        self.lambda_kl = config.gan_config.cvae_config['lambda_kl']
        self.lambda_class = config.gan_config.cvae_config['lambda_class']
        
        # 用于记录训练过程中的损失
        self.loss_history = {
            'recon_loss': [],
            'kl_loss': [],
            'class_loss': []
        }

    def fit(self, dataset):
        """训练CVAE模型
        
        Args:
            dataset: 训练数据集
        """
        # 设置所有模型为训练模式
        self.encoder.train()
        self.generator.train()
        self.classifier.train()

        # 按标签划分训练样本
        self._divide_samples(dataset)

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

        classifier_optimizer = torch.optim.Adam(
            params=self.classifier.parameters(),
            lr=config.gan_config.c_lr,
            betas=(0.5, 0.999),
        )

        # 训练循环
        for e in range(config.gan_config.epochs):
            # 对每个标签类别进行训练
            for target_label in self.samples.keys():
                # 训练分类器多次
                for _ in range(config.gan_config.c_loop_num):
                    classifier_optimizer.zero_grad()
                    
                    # 获取真实样本和标签
                    real_samples = self._get_target_samples(target_label, config.gan_config.batch_size)
                    target_labels_tensor = torch.full([config.gan_config.batch_size], target_label, device=config.device)
                    
                    # 生成假样本用于分类器训练
                    with torch.no_grad():
                        z_prior = torch.randn(config.gan_config.batch_size, config.gan_config.z_size, device=config.device)
                        target_labels_onehot = torch.nn.functional.one_hot(target_labels_tensor, num_classes=self.label_num).float()
                        fake_samples = self.generator(z_prior, target_labels_onehot)
                    
                    # 分别计算真实和生成样本的分类损失
                    real_classifier_output = self.classifier(real_samples)
                    L_class_real = torch.nn.functional.cross_entropy(real_classifier_output, target_labels_tensor)
                    
                    fake_classifier_output = self.classifier(fake_samples)
                    L_class_fake = torch.nn.functional.cross_entropy(fake_classifier_output, target_labels_tensor)
                    
                    # 总分类器损失
                    c_loss = L_class_real + L_class_fake
                    
                    c_loss.backward()
                    classifier_optimizer.step()
                    
                # 训练编码器和生成器多次
                for _ in range(config.gan_config.g_loop_num):
                    encoder_optimizer.zero_grad()
                    generator_optimizer.zero_grad()
                    
                    # 获取真实样本和标签
                    real_samples = self._get_target_samples(target_label, config.gan_config.batch_size)
                    target_labels_tensor = torch.full([config.gan_config.batch_size], target_label, device=config.device)

                    # 从编码器得到潜在向量
                    mu, log_var = self.encoder(real_samples, target_labels_tensor)
                    z_enc = self.encoder.reparameterize(mu, log_var)
                    
                    # 使用编码器得到的潜在向量计算重构样本
                    target_labels_onehot = torch.nn.functional.one_hot(target_labels_tensor, num_classes=self.label_num).float()
                    x_recon = self.generator(z_enc, target_labels_onehot)
                    
                    # 计算各项损失
                    recon_loss = torch.nn.functional.mse_loss(x_recon, real_samples)
                    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / mu.size(0)
                    
                    # 分类损失（使用重构样本）
                    classifier_output = self.classifier(x_recon)
                    class_loss = torch.nn.functional.cross_entropy(classifier_output, target_labels_tensor)
                    
                    # 使用渐进式策略调整分类损失权重
                    if e < 200:  # 前200个epoch，给分类器一些时间学习
                        current_lambda_class = 0.0
                    elif e < 500:
                        progress = (e - 200) / 300
                        current_lambda_class = config.gan_config.cvae_config['lambda_class'] * progress
                    else:
                        current_lambda_class = config.gan_config.cvae_config['lambda_class']
                    
                    # 总损失（只有重构损失、KL散度和分类损失，没有对抗损失）
                    total_loss = (
                        self.lambda_recon * recon_loss +
                        self.lambda_kl * kl_loss +
                        current_lambda_class * class_loss
                    )
                    
                    total_loss.backward()
                    encoder_optimizer.step()
                    generator_optimizer.step()

            # 记录损失值
            self.loss_history['recon_loss'].append(recon_loss.item())
            self.loss_history['kl_loss'].append(kl_loss.item())
            self.loss_history['class_loss'].append(class_loss.item())
            
            # 每50轮打印一次训练进度
            if e % 50 == 0:
                print(f"CVAE训练轮次: {e}/{config.gan_config.epochs}, "
                      f"重构损失: {recon_loss.item():.4f}, "
                      f"KL损失: {kl_loss.item():.4f}, "
                      f"分类损失: {class_loss.item():.4f}")
            
        # 设置所有模型为评估模式
        self.encoder.eval()
        self.generator.eval()
        self.classifier.eval()

    def _divide_samples(self, dataset) -> None:
        """将数据集按标签划分到字典中"""
        for sample, label in dataset:
            label = label.item()
            if label not in self.samples.keys():
                self.samples[label] = sample.unsqueeze(0)
            else:
                self.samples[label] = torch.cat([self.samples[label], sample.unsqueeze(0)])

    def _get_target_samples(self, label: int, num: int) -> torch.Tensor:
        """随机选择指定数量的目标标签样本"""
        available_samples = self.samples[label]
        if len(available_samples) < num:
            # 如果可用样本数量不足，进行有放回采样
            indices = torch.randint(0, len(available_samples), (num,))
            return available_samples[indices]
        elif len(available_samples) == num:
            # 如果可用样本数量刚好等于请求数量，返回所有样本
            return available_samples
        else:
            # 如果可用样本数量大于请求数量，进行无放回采样
            indices = torch.randperm(len(available_samples))[:num]
            return available_samples[indices]

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
        plt.legend()
        
        # 绘制KL损失
        plt.subplot(2, 2, 2)
        plt.plot(self.loss_history['kl_loss'], color='green')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('KL divergence loss')
        plt.legend()
        
        # 绘制分类损失
        plt.subplot(2, 2, 3)
        plt.plot(self.loss_history['class_loss'], color='purple')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Classification Loss')
        plt.legend()
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图片
        loss_plot_path = config.path_config.gan_outs / 'cvae_loss_history.jpg'
        plt.savefig(loss_plot_path)
        plt.close()
        
        print(f"\n损失曲线已保存至: {loss_plot_path}")
        
        # 同时绘制一个综合图
        plt.figure(figsize=(12, 6))
        
        plt.plot(self.loss_history['recon_loss'], label='重构损失', color='blue')
        plt.plot(self.loss_history['kl_loss'], label='KL散度', color='green')
        plt.plot(self.loss_history['class_loss'], label='分类损失', color='purple')
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('CVAE损失曲线')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 保存综合图片
        combined_loss_path = config.path_config.gan_outs / 'cvae_combined_loss.jpg'
        plt.savefig(combined_loss_path)
        plt.close()
        
        print(f"综合损失曲线已保存至: {combined_loss_path}")

    def generate_samples(self, target_label: int, num: int):
        """生成指定标签和数量的样本 - 使用先验分布"""
        condition = torch.full([num], target_label, device=config.device)
        # 将1D标签转换为2D one-hot编码格式
        condition_onehot = torch.nn.functional.one_hot(condition.long(), num_classes=self.label_num).float()
        z_prior = torch.randn(num, config.gan_config.z_size, device=config.device)
        return self.generator(z_prior, condition_onehot).cpu().detach()

    def generate_qualified_samples(self, target_label: int, num: int, confidence_threshold: float = None):
        """生成经过分类器验证的合格样本"""
        result = []
        patience = 20
        # 如果没有提供置信度阈值，则使用配置文件中的值
        if confidence_threshold is None:
            confidence_threshold = config.gan_config.cvae_config['confidence_threshold']
        
        while len(result) < num and patience > 0:
            # 生成样本（使用先验分布）
            samples = self.generate_samples(target_label, min(10, num - len(result)))
            
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

    def reconstruct_samples(self, samples: torch.Tensor, labels: torch.Tensor):
        """重构输入样本 - 使用编码器得到的潜在向量"""
        with torch.no_grad():
            self.encoder.eval()
            self.generator.eval()
            
            samples = samples.to(config.device)
            labels = labels.to(config.device)
            
            # 使用编码器得到潜在向量
            z_enc = self.encoder.encode(samples, labels)
            # 使用生成器重构
            condition_onehot = torch.nn.functional.one_hot(labels.long(), num_classes=self.label_num).float()
            reconstructed = self.generator(z_enc, condition_onehot)
            
            self.encoder.train()
            self.generator.train()
            
            return reconstructed.cpu().detach()