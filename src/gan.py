import pickle
import torch
import src
from src import Classifier, datasets, utils, models, config
import time
import numpy as np
import os


class GAN:
    
    def __init__(self):
        """初始化GAN模型"""
        # 获取数据特征
        self.feature_num = datasets.feature_num
        
        self.generator = models.GANGeneratorModel(
            latent_dim=config.gan_config.z_size,
            output_dim=self.feature_num
        ).to(config.device)
        
        self.discriminator = models.GANDiscriminatorModel(
            in_features=self.feature_num
        ).to(config.device)
        
        self.classifier = models.GANClassifierModel(
            in_features=self.feature_num,
            num_classes=datasets.label_num
        ).to(config.device)
        
        self.samples = None
        
        # 从配置文件加载损失权重
        self.lambda_adv = config.gan_config.gan_config['lambda_adv']
        
        # 用于记录训练过程中的损失
        self.loss_history = {
            'adv_loss': []
        }

    def fit(self, dataset):
        """训练GAN模型
        
        Args:
            dataset: 训练数据集
        """
        # 设置所有模型为训练模式
        self.generator.train()
        self.discriminator.train()
        self.classifier.train()

        # 存储所有训练样本（不按标签分类）
        self._store_samples(dataset)

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
                
            # 训练分类器多次（如果需要的话）
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
                
            # 训练生成器多次
            for _ in range(config.gan_config.g_loop_num):
                generator_optimizer.zero_grad()
                
                # 从先验分布采样
                z_prior = torch.randn(config.gan_config.batch_size, config.gan_config.z_size, device=config.device)
                
                # 生成样本
                generated_samples = self.generator(z_prior)
                
                # 计算对抗损失
                d_fake = self.discriminator(generated_samples)
                adv_loss = -d_fake.mean()
                
                # 总损失
                total_loss = self.lambda_adv * adv_loss
                
                total_loss.backward()
                generator_optimizer.step()

            # 记录损失值
            self.loss_history['adv_loss'].append(adv_loss.item())
            
            # 每50轮打印一次训练进度
            if e % 50 == 0:
                print(f"GAN训练轮次: {e}/{config.gan_config.epochs}, "
                      f"对抗损失: {adv_loss.item():.4f}")
            
        # 设置所有模型为评估模式
        self.generator.eval()
        self.discriminator.eval()
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
        
        plt.figure(figsize=(10, 6))
        
        # 绘制对抗损失
        plt.plot(self.loss_history['adv_loss'], color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Adversarial Loss')
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图片
        loss_plot_path = config.path_config.gan_outs / 'gan_loss_history.jpg'
        plt.savefig(loss_plot_path)
        plt.close()
        
        print(f"\n损失曲线已保存至: {loss_plot_path}")
        
        # 同时绘制一个综合图
        plt.figure(figsize=(12, 6))
        
        # 对抗损失取绝对值
        adv_loss_abs = [abs(loss) for loss in self.loss_history['adv_loss']]
        
        plt.plot(adv_loss_abs, label='对抗损失(绝对值)', color='red')
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('GAN损失曲线')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 保存综合图片
        combined_loss_path = config.path_config.gan_outs / 'gan_combined_loss.jpg'
        plt.savefig(combined_loss_path)
        plt.close()
        
        print(f"综合损失曲线已保存至: {combined_loss_path}")

    def generate_samples(self, num: int):
        """生成指定数量的样本 - 使用先验分布"""
        z_prior = torch.randn(num, config.gan_config.z_size, device=config.device)
        return self.generator(z_prior).cpu().detach()

    def generate_qualified_samples(self, target_label: int, num: int, confidence_threshold: float = None):
        """生成经过分类器验证的合格样本（需要指定目标标签）"""
        result = []
        patience = 20
        # 如果没有提供置信度阈值，则使用配置文件中的值
        if confidence_threshold is None:
            confidence_threshold = config.gan_config.gan_config['confidence_threshold']
        
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