import context
import pickle
import torch
import src
from src import Classifier, datasets, utils, models, config
import time
import numpy as np
import os


class CTGAN:
    """CTGAN模型类，专门为表格数据设计"""
    
    def __init__(self):
        """初始化CTGAN模型"""
        # 获取数据特征和类别信息
        self.feature_num = datasets.feature_num
        self.label_num = datasets.label_num
        
        # 创建模型组件
        self.generator = models.CTGANGeneratorModel(
            latent_dim=config.gan_config.z_size,
            num_classes=self.label_num,
            output_dim=self.feature_num,
            num_columns=self.feature_num  # 假设每列对应一个特征
        ).to(config.device)
        
        self.discriminator = models.CTGANDiscriminatorModel(
            in_features=self.feature_num,
            num_classes=self.label_num,
            num_columns=self.feature_num
        ).to(config.device)
        
        self.classifier = models.CTGANClassifierModel(
            in_features=self.feature_num,
            num_classes=self.label_num
        ).to(config.device)
        
        # 用于存储按标签分类的样本字典
        self.samples = dict()
        
        # 从配置文件加载损失权重
        self.lambda_adv = config.gan_config.ctgan_config['lambda_adv']
        self.lambda_class = config.gan_config.ctgan_config['lambda_class']
        self.lambda_gp = config.gan_config.ctgan_config['lambda_gp']  # 梯度惩罚权重
        
        # 用于记录训练过程中的损失
        self.loss_history = {
            'd_loss': [],
            'g_loss': [],
            'gp_loss': [],
            'class_loss': []
        }

    def fit(self, dataset):
        """训练CTGAN模型
        
        Args:
            dataset: 训练数据集
        """
        # 设置所有模型为训练模式
        self.generator.train()
        self.discriminator.train()
        self.classifier.train()

        # 按标签划分训练样本
        self._divide_samples(dataset)

        # 创建优化器
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
                # 训练判别器多次（CTGAN通常训练5次）
                for _ in range(5):  # 固定为5次，这是WGAN-GP的标准做法
                    discriminator_optimizer.zero_grad()
                    
                    # 获取真实样本和标签
                    real_samples = self._get_target_samples(target_label, config.gan_config.batch_size)
                    target_labels_tensor = torch.full([config.gan_config.batch_size], target_label, device=config.device)
                    
                    # 生成假样本
                    with torch.no_grad():
                        generated_samples = self.generate_samples(target_label, config.gan_config.batch_size)
                    
                    # WGAN-GP损失
                    d_real = self.discriminator(real_samples, target_labels_tensor)
                    d_fake = self.discriminator(generated_samples.detach(), target_labels_tensor)
                    
                    # 计算梯度惩罚
                    gradient_penalty = self.discriminator.calculate_gradient_penalty(
                        real_samples, generated_samples, target_labels_tensor, self.lambda_gp
                    )
                    
                    # 判别器总损失：Wasserstein距离 + 梯度惩罚
                    d_loss = -torch.mean(d_real) + torch.mean(d_fake) + gradient_penalty
                    
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
                        generated_samples = self.generate_samples(target_label, config.gan_config.batch_size)
                    
                    # 分别计算真实和生成样本的分类损失
                    real_classifier_output = self.classifier(real_samples)
                    L_class_real = torch.nn.functional.cross_entropy(real_classifier_output, target_labels_tensor)
                    
                    fake_classifier_output = self.classifier(generated_samples)
                    L_class_fake = torch.nn.functional.cross_entropy(fake_classifier_output, target_labels_tensor)
                    
                    # 总分类器损失
                    c_loss = L_class_real + L_class_fake
                    
                    c_loss.backward()
                    classifier_optimizer.step()
                    
                # 训练生成器（每5次判别器训练后训练1次生成器）
                generator_optimizer.zero_grad()
                
                # 获取目标标签
                target_labels_tensor = torch.full([config.gan_config.batch_size], target_label, device=config.device)
                
                # 生成样本
                generated_samples = self.generate_samples(target_label, config.gan_config.batch_size)
                
                # 生成器损失：负的Wasserstein距离
                d_fake = self.discriminator(generated_samples, target_labels_tensor)
                g_loss = -torch.mean(d_fake)
                
                # 计算分类损失
                classifier_output = self.classifier(generated_samples)
                class_loss = torch.nn.functional.cross_entropy(classifier_output, target_labels_tensor)
                
                # 渐进式策略调整分类损失权重
                if e < 200:
                    current_lambda_class = 0.0
                elif e < 500:
                    progress = (e - 200) / 300
                    current_lambda_class = config.gan_config.ctgan_config['lambda_class'] * progress
                else:
                    current_lambda_class = config.gan_config.ctgan_config['lambda_class']
                
                # 总损失
                total_loss = self.lambda_adv * g_loss + current_lambda_class * class_loss
                
                total_loss.backward()
                generator_optimizer.step()

            # 记录损失值
            self.loss_history['d_loss'].append(d_loss.item())
            self.loss_history['g_loss'].append(g_loss.item())
            self.loss_history['gp_loss'].append(gradient_penalty.item())
            self.loss_history['class_loss'].append(class_loss.item())
            
            # 每50轮打印一次训练进度
            if e % 50 == 0:
                print(f"CTGAN训练轮次: {e}/{config.gan_config.epochs}, "
                      f"判别器损失: {d_loss.item():.4f}, "
                      f"生成器损失: {g_loss.item():.4f}, "
                      f"梯度惩罚: {gradient_penalty.item():.4f}, "
                      f"分类损失: {class_loss.item():.4f}")
            
        # 设置所有模型为评估模式
        self.generator.eval()
        self.discriminator.eval()
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
        # 获取available_samples所在的设备
        device = available_samples.device
        
        if len(available_samples) < num:
            # 如果可用样本数量不足，进行有放回采样
            indices = torch.randint(0, len(available_samples), (num,), device=device)
            return available_samples[indices]
        elif len(available_samples) == num:
            # 如果可用样本数量刚好等于请求数量，返回所有样本
            return available_samples
        else:
            # 如果可用样本数量大于请求数量，进行无放回采样
            indices = torch.randperm(len(available_samples), device=device)[:num]
            return available_samples[indices]

    def plot_loss_history(self):
        """绘制损失历史曲线并保存为图片"""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 8))
        
        # 绘制判别器损失
        plt.subplot(2, 2, 1)
        plt.plot(self.loss_history['d_loss'], color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Discriminator Loss (Wasserstein)')
        plt.legend()
        
        # 绘制生成器损失
        plt.subplot(2, 2, 2)
        plt.plot(self.loss_history['g_loss'], color='blue')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Generator Loss')
        plt.legend()
        
        # 绘制梯度惩罚损失
        plt.subplot(2, 2, 3)
        plt.plot(self.loss_history['gp_loss'], color='green')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Gradient Penalty')
        plt.legend()
        
        # 绘制分类损失
        plt.subplot(2, 2, 4)
        plt.plot(self.loss_history['class_loss'], color='purple')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Classification Loss')
        plt.legend()
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图片
        loss_plot_path = config.path_config.gan_outs / 'ctgan_loss_history.jpg'
        plt.savefig(loss_plot_path)
        plt.close()
        
        print(f"\n损失曲线已保存至: {loss_plot_path}")
        
        # 同时绘制一个综合图
        plt.figure(figsize=(12, 6))
        
        plt.plot(self.loss_history['d_loss'], label='判别器损失', color='red')
        plt.plot(self.loss_history['g_loss'], label='生成器损失', color='blue')
        plt.plot(self.loss_history['gp_loss'], label='梯度惩罚', color='green')
        plt.plot(self.loss_history['class_loss'], label='分类损失', color='purple')
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('CTGAN损失曲线')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 保存综合图片
        combined_loss_path = config.path_config.gan_outs / 'ctgan_combined_loss.jpg'
        plt.savefig(combined_loss_path)
        plt.close()
        
        print(f"综合损失曲线已保存至: {combined_loss_path}")

    def generate_samples(self, target_label: int, num: int, to_cpu: bool = False):
        """生成指定标签和数量的样本"""
        condition = torch.full([num], target_label, device=config.device)
        
        # 生成样本
        samples = self.generator.generate_conditional_samples(num, condition).detach()
        
        # 根据需要移至CPU
        if to_cpu:
            return samples.cpu()
        return samples

    def generate_qualified_samples(self, target_label: int, num: int, confidence_threshold: float = None):
        """生成经过分类器验证的合格样本"""
        result = []
        patience = 20
        # 如果没有提供置信度阈值，则使用配置文件中的值
        if confidence_threshold is None:
            confidence_threshold = config.gan_config.ctgan_config['confidence_threshold']
        
        while len(result) < num and patience > 0:
            # 生成样本（现在在GPU上）
            samples = self.generate_samples(target_label, min(10, num - len(result)))
            
            # 使用分类器验证
            with torch.no_grad():
                self.classifier.eval()
                # samples已经在GPU上，不需要再次移动
                classifier_output = self.classifier(samples)
                self.classifier.train()
                
                # 获取预测概率
                probs = torch.softmax(classifier_output, dim=1)
                max_probs, preds = torch.max(probs, dim=1)
                
                # 筛选高置信度且类别正确的样本
                valid_mask = (max_probs > confidence_threshold) & (preds == target_label)
                # 保持mask在GPU上，因为samples在GPU上
                valid_samples = samples[valid_mask]
                # 将有效样本移到CPU上，加入结果列表
                result.extend(valid_samples.cpu())
                
                if len(valid_samples) == 0:
                    patience -= 1
        
        return torch.stack(result) if result else torch.tensor([])
    
    def calculate_metrics(self, real_samples: torch.Tensor, fake_samples: torch.Tensor):
        """计算生成样本的评估指标"""
        metrics = {}
        
        with torch.no_grad():
            # 1. 计算均值差异
            real_mean = torch.mean(real_samples, dim=0)
            fake_mean = torch.mean(fake_samples, dim=0)
            metrics['mean_difference'] = torch.norm(real_mean - fake_mean).item()
            
            # 2. 计算协方差矩阵差异
            real_cov = torch.cov(real_samples.T)
            fake_cov = torch.cov(fake_samples.T)
            metrics['cov_difference'] = torch.norm(real_cov - fake_cov).item()
            
            # 3. 计算最大均值差异（MMD）近似
            # 使用RBF核的近似计算
            def rbf_kernel(x, y, sigma=1.0):
                x_size = x.size(0)
                y_size = y.size(0)
                dim = x.size(1)
                
                x = x.view(x_size, 1, dim)
                y = y.view(1, y_size, dim)
                
                return torch.exp(-torch.sum((x - y) ** 2, dim=2) / (2 * sigma ** 2))
            
            # 采样部分数据计算MMD
            n_samples = min(100, real_samples.size(0), fake_samples.size(0))
            real_subsample = real_samples[:n_samples]
            fake_subsample = fake_samples[:n_samples]
            
            k_real_real = rbf_kernel(real_subsample, real_subsample).mean()
            k_fake_fake = rbf_kernel(fake_subsample, fake_subsample).mean()
            k_real_fake = rbf_kernel(real_subsample, fake_subsample).mean()
            
            metrics['mmd_approx'] = (k_real_real + k_fake_fake - 2 * k_real_fake).item()
        
        return metrics