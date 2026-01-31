import context
import pickle
import torch
import src
from src import Classifier, datasets, utils, models, config
from src.models.qg_smote_models import QuantileEncoderModel, QGGeneratorModel, QGDiscriminatorModel, QuantileRegressorModel, QGClassifierModel
import time
import numpy as np
import os


class QG_SMOTE:
    
    def __init__(self):

        self.feature_num = datasets.feature_num
        self.label_num = datasets.label_num
        
        # 创建模型组件
        self.encoder = QuantileEncoderModel(
            input_dim=self.feature_num,
            num_classes=self.label_num,
            latent_dim=config.gan_config.z_size,
            num_quantiles=config.gan_config.qg_smote_config['num_quantiles']
        ).to(config.device)
        
        self.generator = QGGeneratorModel(
            latent_dim=config.gan_config.z_size,
            num_classes=self.label_num,
            output_dim=self.feature_num
        ).to(config.device)
        
        self.discriminator = QGDiscriminatorModel(
            in_features=self.feature_num,
            num_classes=self.label_num
        ).to(config.device)
        
        self.quantile_regressor = QuantileRegressorModel(
            in_features=self.feature_num,
            num_classes=self.label_num,
            num_quantiles=config.gan_config.qg_smote_config['num_quantiles']
        ).to(config.device)
        
        self.classifier = QGClassifierModel(
            in_features=self.feature_num,
            num_classes=self.label_num
        ).to(config.device)
        
        # 用于存储按标签分类的样本字典
        self.samples = dict()
        
        # 从配置文件加载损失权重
        self.lambda_recon = config.gan_config.qg_smote_config['lambda_recon']
        self.lambda_quantile = config.gan_config.qg_smote_config['lambda_quantile']
        self.lambda_adv = config.gan_config.qg_smote_config['lambda_adv']
        self.lambda_class = config.gan_config.qg_smote_config['lambda_class']
        
        # 用于记录训练过程中的损失
        self.loss_history = {
            'recon_loss': [],
            'quantile_loss': [],
            'adv_loss': [],
            'class_loss': []
        }

    def fit(self, dataset):
        """训练QG-SMOTE模型
        
        Args:
            dataset: 训练数据集
        """
        # 设置所有模型为训练模式
        self.encoder.train()
        self.generator.train()
        self.discriminator.train()
        self.quantile_regressor.train()
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
        
        quantile_optimizer = torch.optim.Adam(
            params=self.quantile_regressor.parameters(),
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
                # 训练判别器多次
                for _ in range(config.gan_config.d_loop_num):
                    discriminator_optimizer.zero_grad()
                    
                    # 获取真实样本和标签
                    real_samples = self._get_target_samples(target_label, config.gan_config.batch_size)
                    target_labels_tensor = torch.full([config.gan_config.batch_size], target_label, device=config.device)
                    
                    # 从分位数编码器采样生成样本
                    with torch.no_grad():
                        # 获取分位数
                        quantiles = self.encoder(real_samples, target_labels_tensor)
                        # 从分位数中采样
                        z_quantile = self.encoder.sample_from_quantiles(quantiles)
                        target_labels_onehot = torch.nn.functional.one_hot(target_labels_tensor, num_classes=self.label_num).float()
                        generated_samples = self.generator(z_quantile, target_labels_onehot)
                    
                    # 判别器对真实样本的输出
                    d_real = self.discriminator(real_samples, target_labels_tensor)
                    d_real_loss = -d_real.mean()
                    
                    # 判别器对生成样本的输出
                    d_fake = self.discriminator(generated_samples.detach(), target_labels_tensor)
                    d_fake_loss = d_fake.mean()

                    # 判别器总损失
                    d_loss = d_real_loss + d_fake_loss
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
                        quantiles = self.encoder(real_samples, target_labels_tensor)
                        z_quantile = self.encoder.sample_from_quantiles(quantiles)
                        target_labels_onehot = torch.nn.functional.one_hot(target_labels_tensor, num_classes=self.label_num).float()
                        fake_samples = self.generator(z_quantile, target_labels_onehot)
                    
                    # 分别计算真实和生成样本的分类损失
                    real_classifier_output = self.classifier(real_samples)
                    L_class_real = torch.nn.functional.cross_entropy(real_classifier_output, target_labels_tensor)
                    
                    fake_classifier_output = self.classifier(fake_samples)
                    L_class_fake = torch.nn.functional.cross_entropy(fake_classifier_output, target_labels_tensor)
                    
                    # 总分类器损失
                    c_loss = L_class_real + L_class_fake
                    
                    c_loss.backward()
                    classifier_optimizer.step()
                    
                # 训练分位数回归器
                for _ in range(1):  # 通常只需要训练一次
                    quantile_optimizer.zero_grad()
                    
                    # 获取真实样本和标签
                    real_samples = self._get_target_samples(target_label, config.gan_config.batch_size)
                    target_labels_tensor = torch.full([config.gan_config.batch_size], target_label, device=config.device)
                    
                    # 预测分位数
                    pred_quantiles = self.quantile_regressor(real_samples, target_labels_tensor)
                    
                    # 分位数回归损失（分位数损失）
                    quantile_loss = self._quantile_loss(real_samples, pred_quantiles)
                    
                    quantile_loss.backward()
                    quantile_optimizer.step()
                    
                # 训练编码器和生成器多次
                for _ in range(config.gan_config.g_loop_num):
                    encoder_optimizer.zero_grad()
                    generator_optimizer.zero_grad()
                    
                    # 获取真实样本和标签
                    real_samples = self._get_target_samples(target_label, config.gan_config.batch_size)
                    target_labels_tensor = torch.full([config.gan_config.batch_size], target_label, device=config.device)

                    # 1. 从编码器得到分位数
                    quantiles = self.encoder(real_samples, target_labels_tensor)
                    
                    # 2. 从分位数中采样潜在向量
                    z_quantile = self.encoder.sample_from_quantiles(quantiles)
                    
                    # 3. 使用采样的潜在向量计算重构样本
                    target_labels_onehot = torch.nn.functional.one_hot(target_labels_tensor, num_classes=self.label_num).float()
                    x_recon = self.generator(z_quantile, target_labels_onehot)
                    
                    # 4. 从先验分布采样生成样本（用于对抗损失）
                    z_prior = torch.randn(config.gan_config.batch_size, config.gan_config.z_size, device=config.device)
                    x_fake = self.generator(z_prior, target_labels_onehot)
                    
                    # 计算各项损失
                    # 重构损失
                    recon_loss = torch.nn.functional.mse_loss(x_recon, real_samples)
                    
                    # 分位数损失（暂时移除，因为 quantile_regressor 预测的是特征空间的分位数）
                    quantile_loss = torch.tensor(0.0, device=config.device)
                    
                    # 对抗损失
                    d_fake = self.discriminator(x_fake, target_labels_tensor)
                    adv_loss = -d_fake.mean()
                    
                    # 分类损失
                    classifier_fake_output = self.classifier(x_fake)
                    class_loss = torch.nn.functional.cross_entropy(classifier_fake_output, target_labels_tensor)

                    # 渐进式策略调整分类损失权重
                    if e < 200:
                        current_lambda_class = 0.0
                    elif e < 500:
                        progress = (e - 200) / 300
                        current_lambda_class = config.gan_config.qg_smote_config['lambda_class'] * progress
                    else:
                        current_lambda_class = config.gan_config.qg_smote_config['lambda_class']
                    
                    # 总损失
                    total_loss = (
                        self.lambda_recon * recon_loss +
                        self.lambda_quantile * quantile_loss +
                        self.lambda_adv * adv_loss +
                        current_lambda_class * class_loss
                    )
                    
                    total_loss.backward()
                    encoder_optimizer.step()
                    generator_optimizer.step()

            # 记录损失值
            self.loss_history['recon_loss'].append(recon_loss.item())
            self.loss_history['quantile_loss'].append(quantile_loss.item())
            self.loss_history['adv_loss'].append(adv_loss.item())
            self.loss_history['class_loss'].append(class_loss.item())
            
            # 每50轮打印一次训练进度
            if e % 50 == 0:
                print(f"QG-SMOTE训练轮次: {e}/{config.gan_config.epochs}, "
                      f"重构损失: {recon_loss.item():.4f}, "
                      f"分位数损失: {quantile_loss.item():.4f}, "
                      f"对抗损失: {adv_loss.item():.4f}, "
                      f"分类损失: {class_loss.item():.4f}")
            
        # 设置所有模型为评估模式
        self.encoder.eval()
        self.generator.eval()
        self.discriminator.eval()
        self.quantile_regressor.eval()
        self.classifier.eval()

    def _quantile_loss(self, x: torch.Tensor, quantiles: torch.Tensor, quantile_levels: list = None):
        """
        计算分位数回归损失
        
        Args:
            x: 真实值 [batch_size, feature_dim]
            quantiles: 预测的分位数 [batch_size, feature_dim, num_quantiles]
            quantile_levels: 分位数水平列表，默认为[0.25, 0.5, 0.75]
            
        Returns:
            分位数损失
        """
        if quantile_levels is None:
            quantile_levels = [0.25, 0.5, 0.75]
        
        # 将分位数水平转换为张量
        quantile_levels_tensor = torch.tensor(quantile_levels, device=x.device).view(1, 1, -1)
        
        # 计算分位数损失
        errors = x.unsqueeze(-1) - quantiles  # [batch_size, feature_dim, num_quantiles]
        loss = torch.max(quantile_levels_tensor * errors, (quantile_levels_tensor - 1) * errors)
        
        return loss.mean()

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
        
        # 绘制分位数损失
        plt.subplot(2, 2, 2)
        plt.plot(self.loss_history['quantile_loss'], color='orange')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Quantile Loss')
        plt.legend()
        
        # 绘制对抗损失
        plt.subplot(2, 2, 3)
        plt.plot(self.loss_history['adv_loss'], color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Adversarial Loss')
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
        loss_plot_path = config.path_config.gan_outs / 'qg_smote_loss_history.jpg'
        plt.savefig(loss_plot_path)
        plt.close()
        
        print(f"\n损失曲线已保存至: {loss_plot_path}")
        
        # 同时绘制一个综合图
        plt.figure(figsize=(12, 6))
        
        # 对抗损失取绝对值
        adv_loss_abs = [abs(loss) for loss in self.loss_history['adv_loss']]
        
        plt.plot(self.loss_history['recon_loss'], label='重构损失', color='blue')
        plt.plot(self.loss_history['quantile_loss'], label='分位数损失', color='orange')
        plt.plot(adv_loss_abs, label='对抗损失(绝对值)', color='red')
        plt.plot(self.loss_history['class_loss'], label='分类损失', color='purple')
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('QG-SMOTE损失曲线')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 保存综合图片
        combined_loss_path = config.path_config.gan_outs / 'qg_smote_combined_loss.jpg'
        plt.savefig(combined_loss_path)
        plt.close()
        
        print(f"综合损失曲线已保存至: {combined_loss_path}")

    def generate_samples(self, target_label: int, num: int, method: str = 'quantile'):
        """
        生成指定标签和数量的样本
        
        Args:
            target_label: 目标标签
            num: 生成样本数量
            method: 生成方法，可选 'quantile'（分位数采样）或 'prior'（先验采样）
            
        Returns:
            生成的样本
        """
        condition = torch.full([num], target_label, device=config.device)
        condition_onehot = torch.nn.functional.one_hot(condition.long(), num_classes=self.label_num).float()
        
        if method == 'quantile':
            # 使用分位数采样生成样本
            # 需要先生成一些真实样本来获取分位数
            real_samples = self._get_target_samples(target_label, min(num, len(self.samples[target_label])))
            if len(real_samples) < num:
                # 如果需要更多样本，重复采样
                repeat_times = (num + len(real_samples) - 1) // len(real_samples)
                real_samples = real_samples.repeat(repeat_times, 1)[:num]
            
            # 获取分位数
            condition_tensor = torch.full([len(real_samples)], target_label, device=config.device)
            quantiles = self.encoder(real_samples, condition_tensor)
            
            # 从分位数中采样
            z = self.encoder.sample_from_quantiles(quantiles)
        else:
            # 使用先验分布
            z = torch.randn(num, config.gan_config.z_size, device=config.device)
        
        return self.generator(z, condition_onehot).cpu().detach()

    def generate_qualified_samples(self, target_label: int, num: int, 
                                  confidence_threshold: float = None, method: str = 'quantile'):
        """生成经过分类器验证的合格样本"""
        result = []
        patience = 20
        # 如果没有提供置信度阈值，则使用配置文件中的值
        if confidence_threshold is None:
            confidence_threshold = config.gan_config.qg_smote_config['confidence_threshold']
        
        while len(result) < num and patience > 0:
            # 生成样本
            samples = self.generate_samples(target_label, min(10, num - len(result)), method=method)
            
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

    def analyze_quantiles(self, target_label: int, num_samples: int = 100):
        """分析特定类别的分位数分布"""
        with torch.no_grad():
            # 获取真实样本
            real_samples = self._get_target_samples(target_label, min(num_samples, len(self.samples[target_label])))
            condition = torch.full([len(real_samples)], target_label, device=config.device)
            
            # 获取分位数
            quantiles = self.encoder(real_samples, condition)
            
            # 计算分位数统计
            quantile_stats = {
                'mean': torch.mean(quantiles, dim=0).cpu().numpy(),
                'std': torch.std(quantiles, dim=0).cpu().numpy(),
                'min': torch.min(quantiles, dim=0)[0].cpu().numpy(),
                'max': torch.max(quantiles, dim=0)[0].cpu().numpy(),
            }
            
            return quantile_stats