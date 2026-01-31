import context
import pickle
import torch
import src
from src import Classifier, datasets, utils, models, config
from src.models.rain_gan_models import RAINEncoderModel, RAINGeneratorModel, RAINDiscriminatorModel, RAINClassifierModel
import time
import numpy as np
import os


class RAIN_GAN:
    
    def __init__(self):
        self.feature_num = datasets.feature_num
        self.label_num = datasets.label_num
        
        # 创建模型组件
        self.encoder = RAINEncoderModel(
            input_dim=self.feature_num,
            num_classes=self.label_num,
            latent_dim=config.gan_config.z_size
        ).to(config.device)
        
        self.generator = RAINGeneratorModel(
            latent_dim=config.gan_config.z_size,
            num_classes=self.label_num,
            output_dim=self.feature_num
        ).to(config.device)
        
        self.discriminator = RAINDiscriminatorModel(
            in_features=self.feature_num,
            num_classes=self.label_num
        ).to(config.device)
        
        self.classifier = RAINClassifierModel(
            in_features=self.feature_num,
            num_classes=self.label_num
        ).to(config.device)
        
        # 用于存储按标签分类的样本字典
        self.samples = dict()
        
        # 从配置文件加载损失权重
        self.lambda_recon = config.gan_config.rain_gan_config['lambda_recon']
        self.lambda_kl = config.gan_config.rain_gan_config['lambda_kl']
        self.lambda_adv = config.gan_config.rain_gan_config['lambda_adv']
        self.lambda_class = config.gan_config.rain_gan_config['lambda_class']
        self.lambda_attention = config.gan_config.rain_gan_config['lambda_attention']
        
        # 用于记录训练过程中的损失
        self.loss_history = {
            'recon_loss': [],
            'kl_loss': [],
            'adv_loss': [],
            'class_loss': [],
            'attention_loss': []
        }
        
        # 注意力权重历史记录
        self.attention_history = {
            'encoder': [],
            'generator': [],
            'discriminator': [],
            'classifier': []
        }

    def fit(self, dataset):
        """训练RAIN-GAN模型
        
        Args:
            dataset: 训练数据集
        """
        # 设置所有模型为训练模式
        self.encoder.train()
        self.generator.train()
        self.discriminator.train()
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
        
        discriminator_optimizer = torch.optim.Adam(
            params=self.discriminator.parameters(),
            lr=config.gan_config.d_lr,
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
                # 训练判别器多次
                for _ in range(config.gan_config.d_loop_num):
                    discriminator_optimizer.zero_grad()
                    
                    # 获取真实样本和标签
                    real_samples = self._get_target_samples(target_label, config.gan_config.batch_size)
                    target_labels_tensor = torch.full([config.gan_config.batch_size], target_label, device=config.device)
                    target_labels_onehot = torch.nn.functional.one_hot(target_labels_tensor, num_classes=self.label_num).float()
                    
                    # 从先验分布采样生成样本
                    with torch.no_grad():
                        z_prior = torch.randn(config.gan_config.batch_size, config.gan_config.z_size, device=config.device)
                        generated_samples = self.generator.generate_conditional_samples(
                            config.gan_config.batch_size, target_labels_tensor
                        )
                    
                    # 判别器对真实样本的输出
                    d_real = self.discriminator(real_samples, target_labels_tensor)
                    d_real_loss = -d_real.mean()
                    
                    # 判别器对生成样本的输出
                    d_fake = self.discriminator(generated_samples.detach(), target_labels_tensor)
                    d_fake_loss = d_fake.mean()

                    # 注意力一致性损失
                    attention_loss_d = self._calculate_attention_loss('discriminator')
                    
                    # 判别器总损失
                    d_loss = d_real_loss + d_fake_loss + self.lambda_attention * attention_loss_d
                    d_loss.backward()
                    discriminator_optimizer.step()

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
                        fake_samples = self.generator.generate_conditional_samples(
                            config.gan_config.batch_size, target_labels_tensor
                        )
                    
                    # 分别计算真实和生成样本的分类损失
                    real_classifier_output = self.classifier(real_samples)
                    L_class_real = torch.nn.functional.cross_entropy(real_classifier_output, target_labels_tensor)
                    
                    fake_classifier_output = self.classifier(fake_samples)
                    L_class_fake = torch.nn.functional.cross_entropy(fake_classifier_output, target_labels_tensor)
                    
                    # 注意力一致性损失
                    attention_loss_c = self._calculate_attention_loss('classifier')
                    
                    # 总分类器损失
                    c_loss = L_class_real + L_class_fake + self.lambda_attention * attention_loss_c
                    
                    c_loss.backward()
                    classifier_optimizer.step()
                    
                # 训练编码器和生成器多次
                for _ in range(config.gan_config.g_loop_num):
                    encoder_optimizer.zero_grad()
                    generator_optimizer.zero_grad()
                    
                    # 获取真实样本和标签
                    real_samples = self._get_target_samples(target_label, config.gan_config.batch_size)
                    target_labels_tensor = torch.full([config.gan_config.batch_size], target_label, device=config.device)

                    # 1. 从编码器得到 z_enc（用于重构和KL损失）
                    mu, log_var = self.encoder(real_samples, target_labels_tensor)
                    z_enc = self.encoder.reparameterize(mu, log_var)
                    
                    # 2. 从先验分布采样 z_prior（用于对抗和分类损失）
                    z_prior = torch.randn(config.gan_config.batch_size, config.gan_config.z_size, device=config.device)
                    
                    # 3. 使用 z_enc 计算重构样本（用于重构和KL损失）
                    target_labels_onehot = torch.nn.functional.one_hot(target_labels_tensor, num_classes=self.label_num).float()
                    x_recon = self.generator.generate_conditional_samples(
                        config.gan_config.batch_size, target_labels_tensor
                    )
                    
                    # 4. 使用 z_prior 计算生成样本（用于对抗和分类损失）
                    x_fake = self.generator.generate_conditional_samples(
                        config.gan_config.batch_size, target_labels_tensor
                    )
                    
                    # 计算各项损失
                    # 重构损失和KL损失使用编码器得到的 z_enc
                    recon_loss = torch.nn.functional.mse_loss(x_recon, real_samples)
                    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / mu.size(0)
                    
                    # 对抗损失使用判别器，分类损失使用分类器
                    d_fake = self.discriminator(x_fake, target_labels_tensor)
                    adv_loss = -d_fake.mean()

                    # 分类损失
                    classifier_fake_output = self.classifier(x_fake)
                    class_loss = torch.nn.functional.cross_entropy(classifier_fake_output, target_labels_tensor)

                    # 注意力一致性损失
                    attention_loss_e = self._calculate_attention_loss('encoder')
                    attention_loss_g = self._calculate_attention_loss('generator')
                    attention_loss_total = attention_loss_e + attention_loss_g

                    # 在生成器训练部分修改分类损失计算
                    if e < 200:
                        current_lambda_class = 0.0
                    elif e < 500:
                        progress = (e - 200) / 300
                        current_lambda_class = config.gan_config.rain_gan_config['lambda_class'] * progress
                    else:
                        current_lambda_class = config.gan_config.rain_gan_config['lambda_class']
                    
                    # 总损失
                    total_loss = (
                        self.lambda_recon * recon_loss +
                        self.lambda_kl * kl_loss +
                        self.lambda_adv * adv_loss +
                        current_lambda_class * class_loss +
                        self.lambda_attention * attention_loss_total
                    )
                    
                    total_loss.backward()
                    encoder_optimizer.step()
                    generator_optimizer.step()
                    
                    # 记录注意力权重
                    if e % 50 == 0:
                        self._record_attention_weights()

            # 记录损失值
            self.loss_history['recon_loss'].append(recon_loss.item())
            self.loss_history['kl_loss'].append(kl_loss.item())
            self.loss_history['adv_loss'].append(adv_loss.item())
            self.loss_history['class_loss'].append(class_loss.item())
            self.loss_history['attention_loss'].append(attention_loss_total.item())
            
            # 每10轮打印一次训练进度
            if e % 50 == 0:
                print(f"RAIN-GAN训练轮次: {e}/{config.gan_config.epochs}, "
                      f"重构损失: {recon_loss.item():.4f}, "
                      f"KL损失: {kl_loss.item():.4f}, "
                      f"对抗损失: {adv_loss.item():.4f}, "
                      f"分类损失: {class_loss.item():.4f}, "
                      f"注意力损失: {attention_loss_total.item():.4f}")
            
        # 设置所有模型为评估模式
        self.encoder.eval()
        self.generator.eval()
        self.discriminator.eval()
        self.classifier.eval()

    def _calculate_attention_loss(self, model_type: str) -> torch.Tensor:
        """计算注意力一致性损失"""
        attention_weights = None
        
        if model_type == 'encoder':
            attention_weights = self.encoder.attention_weights
        elif model_type == 'generator':
            attention_weights = self.generator.attention_weights
        elif model_type == 'discriminator':
            attention_weights = self.discriminator.attention_weights
        elif model_type == 'classifier':
            attention_weights = self.classifier.attention_weights
        
        if attention_weights is None:
            return torch.tensor(0.0, device=config.device)
        
        # 计算注意力分布的熵（鼓励注意力集中）
        entropy = -torch.sum(attention_weights * torch.log(attention_weights + 1e-8), dim=-1)
        attention_loss = torch.mean(entropy)
        
        return attention_loss
    
    def _record_attention_weights(self):
        """记录注意力权重（用于分析）"""
        with torch.no_grad():
            # 记录编码器注意力权重（如果可用）
            if hasattr(self.encoder, 'attention_weights') and self.encoder.attention_weights is not None:
                self.attention_history['encoder'].append(
                    self.encoder.attention_weights.mean().item()
                )
            
            # 记录生成器注意力权重
            if hasattr(self.generator, 'attention_weights') and self.generator.attention_weights is not None:
                self.attention_history['generator'].append(
                    self.generator.attention_weights.mean().item()
                )
            
            # 记录判别器注意力权重
            if hasattr(self.discriminator, 'attention_weights') and self.discriminator.attention_weights is not None:
                self.attention_history['discriminator'].append(
                    self.discriminator.attention_weights.mean().item()
                )
            
            # 记录分类器注意力权重
            if hasattr(self.classifier, 'attention_weights') and self.classifier.attention_weights is not None:
                self.attention_history['classifier'].append(
                    self.classifier.attention_weights.mean().item()
                )

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
            indices = torch.randint(0, len(available_samples), (num,))
            return available_samples[indices]
        elif len(available_samples) == num:
            return available_samples
        else:
            indices = torch.randperm(len(available_samples))[:num]
            return available_samples[indices]

    def plot_loss_history(self):
        """绘制损失历史曲线并保存为图片"""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(15, 10))
        
        # 绘制重构损失
        plt.subplot(3, 2, 1)
        plt.plot(self.loss_history['recon_loss'], color='blue')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Reconstruction Loss')
        plt.legend()
        
        # 绘制KL损失
        plt.subplot(3, 2, 2)
        plt.plot(self.loss_history['kl_loss'], color='green')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('KL divergence loss')
        plt.legend()
        
        # 绘制对抗损失
        plt.subplot(3, 2, 3)
        plt.plot(self.loss_history['adv_loss'], color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Adversarial Loss')
        plt.legend()
        
        # 绘制分类损失
        plt.subplot(3, 2, 4)
        plt.plot(self.loss_history['class_loss'], color='purple')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Classification Loss')
        plt.legend()
        
        # 绘制注意力损失
        plt.subplot(3, 2, 5)
        plt.plot(self.loss_history['attention_loss'], color='orange')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Attention Loss')
        plt.legend()
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图片
        loss_plot_path = config.path_config.gan_outs / 'rain_gan_loss_history.jpg'
        plt.savefig(loss_plot_path)
        plt.close()
        
        print(f"\n损失曲线已保存至: {loss_plot_path}")
        
        # 绘制注意力权重历史
        if len(self.attention_history['encoder']) > 0:
            plt.figure(figsize=(12, 8))
            
            # 绘制各个模型的注意力权重
            for model_name, weights in self.attention_history.items():
                if len(weights) > 0:
                    plt.plot(weights, label=f'{model_name}注意力权重')
            
            plt.xlabel('Epoch (每50轮记录一次)')
            plt.ylabel('平均注意力权重')
            plt.title('RAIN-GAN注意力权重历史')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 保存图片
            attention_plot_path = config.path_config.gan_outs / 'rain_gan_attention_history.jpg'
            plt.savefig(attention_plot_path)
            plt.close()
            
            print(f"注意力权重曲线已保存至: {attention_plot_path}")

    def generate_samples(self, target_label: int, num: int):
        """生成指定标签和数量的样本 - 使用先验分布"""
        condition = torch.full([num], target_label, device=config.device)
        return self.generator.generate_conditional_samples(num, condition).cpu().detach()

    def generate_qualified_samples(self, target_label: int, num: int, confidence_threshold: float = None):
        """生成经过分类器验证的合格样本"""
        result = []
        patience = 20
        # 如果没有提供置信度阈值，则使用配置文件中的值
        if confidence_threshold is None:
            confidence_threshold = config.gan_config.rain_gan_config['confidence_threshold']
        
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
            
            # 将潜在向量与条件拼接
            condition_onehot = torch.nn.functional.one_hot(labels.long(), num_classes=self.label_num).float()
            z_cond = torch.cat([z_enc, condition_onehot], dim=1)
            
            # 添加序列维度
            z_cond = z_cond.unsqueeze(1)
            
            # 使用生成器重构
            reconstructed = self.generator(z_cond).squeeze(1)
            
            self.encoder.train()
            self.generator.train()
            
            return reconstructed.cpu().detach()
    
    def visualize_attention(self, samples: torch.Tensor, labels: torch.Tensor):
        """可视化注意力权重"""
        with torch.no_grad():
            self.encoder.eval()
            self.classifier.eval()
            
            samples = samples.to(config.device)
            labels = labels.to(config.device)
            
            # 获取编码器注意力
            _ = self.encoder(samples, labels)
            encoder_attention = self.encoder.attention_weights
            
            # 获取分类器注意力
            _ = self.classifier(samples)
            classifier_attention = self.classifier.attention_weights
            
            return {
                'encoder_attention': encoder_attention.cpu().numpy() if encoder_attention is not None else None,
                'classifier_attention': classifier_attention.cpu().numpy() if classifier_attention is not None else None
            }