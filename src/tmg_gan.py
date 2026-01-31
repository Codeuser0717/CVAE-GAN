
import random

import torch
from torch.nn.functional import cross_entropy, cosine_similarity
from matplotlib import pyplot as plt
from src import models, config, datasets


class TMGGAN:
    """Targeted Multi-Generator GAN模型类"""

    def __init__(self):
        """初始化TMG-GAN模型"""
        self.cd = models.TMGGANCDModel(datasets.feature_num, datasets.label_num).to(config.device)
        # 创建多个生成器列表，每个标签对应一个生成器
        self.generators = [
            models.TMGGANGeneratorModel(config.gan_config.z_size, datasets.feature_num).to(config.device)
            for _ in range(datasets.label_num)
        ]
        # 用于存储按标签分类的样本字典
        self.samples = dict()

    def fit(self, dataset):
        """训练TMG-GAN模型
        
        Args:
            dataset: 训练数据集
        """
        # 设置CD模型为训练模式
        self.cd.train()
        # 设置所有生成器为训练模式
        for i in self.generators:
            i.train()

        # 按标签划分训练样本
        self._divide_samples(dataset)
        # 创建CD模型的优化器
        cd_optimizer = torch.optim.Adam(
            params=self.cd.parameters(),  # CD模型的参数
            lr=config.gan_config.c_lr,  # 学习率
            betas=(0.5, 0.999),  # Adam优化器的超参数
        )
        # 为每个生成器创建优化器
        g_optimizers = [
            torch.optim.Adam(
                params=self.generators[i].parameters(),  # 第i个生成器的参数
                lr=config.gan_config.g_lr,  # 学习率
                betas=(0.5, 0.999),  # Adam优化器的超参数
            )
            for i in range(datasets.label_num)
        ]
        # 训练循环，迭代指定的轮数
        for e in range(config.gan_config.epochs):
            # 打印训练进度百分比
            print(f'\r{(e + 1) / config.gan_config.epochs: .2%}', end='')

            # 对每个标签类别进行训练
            for target_label in self.samples.keys():
                # 训练分类器-判别器(CD)多次
                for _ in range(config.gan_config.c_loop_num):
                    # 清空CD优化器的梯度
                    cd_optimizer.zero_grad()
                    # 获取目标标签的真实样本
                    real_samples = self._get_target_samples(target_label, config.gan_config.batch_size)
                    # 将真实样本输入CD模型，获取判别分数和预测标签
                    score_real, predicted_labels = self.cd(real_samples)
                    # 计算真实样本的平均判别分数
                    score_real = score_real.mean()
                    # 使用对应标签的生成器生成样本
                    generated_samples = self.generators[target_label].generate_samples(config.gan_config.batch_size)
                    # 将生成样本输入CD模型，获取判别分数
                    score_generated = self.cd(generated_samples)[0].mean()
                    # 计算判别器损失
                    d_loss = (score_generated - score_real) / 2
                    # 计算分类器损失（交叉熵损失）
                    c_loss = cross_entropy(
                        input=predicted_labels,
                        target=torch.full([len(predicted_labels)], target_label, device=config.device),
                    )
                    # 合并损失
                    loss = d_loss + c_loss
                    # 反向传播计算梯度
                    loss.backward()
                    # 更新CD模型参数
                    cd_optimizer.step()

                # 训练生成器多次
                for _ in range(config.gan_config.g_loop_num):
                    # 清空当前标签生成器优化器的梯度
                    g_optimizers[target_label].zero_grad()
                    # 生成样本
                    generated_samples = self.generators[target_label].generate_samples(config.gan_config.batch_size)
                    # 获取真实样本
                    real_samples = self._get_target_samples(target_label, config.gan_config.batch_size)
                    # 将真实样本输入CD模型以获取隐藏层状态
                    self.cd(real_samples)
                    hidden_real = self.cd.hidden_status
                    # 将生成样本输入CD模型
                    score_generated, predicted_labels = self.cd(generated_samples)
                    hidden_generated = self.cd.hidden_status
                    # 计算隐藏层状态的余弦相似度损失（鼓励生成样本的特征分布接近真实样本）
                    cd_hidden_loss = - cosine_similarity(hidden_real, hidden_generated).mean()
                    # 计算生成样本的平均判别分数
                    score_generated = score_generated.mean()
                    # 计算生成样本的分类损失
                    loss_label = cross_entropy(
                        input=predicted_labels,
                        target=torch.full([len(predicted_labels)], target_label, device=config.device),
                    )
                    # 在早期训练阶段（前1000轮）不使用隐藏层损失
                    if e < 1000:
                        cd_hidden_loss = 0
                    # 合并生成器的总损失
                    g_loss = -score_generated + loss_label + cd_hidden_loss
                    # 反向传播计算梯度
                    g_loss.backward()
                    # 更新生成器参数
                    g_optimizers[target_label].step()
            # 清空所有生成器优化器的梯度，为计算生成器间隐藏状态差异做准备
            for i in g_optimizers:
                i.zero_grad()
            # 让每个生成器生成一些样本，以获取它们的隐藏层状态
            for i in self.generators:
                i.generate_samples(3)
            # 存储不同生成器之间隐藏层状态的余弦相似度
            g_hidden_losses = []
            # 计算所有不同生成器对之间的隐藏层状态余弦相似度
            for i, _ in enumerate(self.generators):
                for j, _ in enumerate(self.generators):
                    if i == j:  # 跳过相同生成器
                        continue
                    else:
                        g_hidden_losses.append(
                            cosine_similarity(
                                self.generators[i].hidden_status,
                                self.generators[j].hidden_status,
                            )
                        )
            # 计算生成器间隐藏层状态的平均相似度损失（除以特征数以归一化）
            g_hidden_loss = torch.mean(torch.stack(g_hidden_losses)) / datasets.feature_num
            # 反向传播计算梯度
            g_hidden_loss.backward()
            # 更新所有生成器参数，鼓励不同生成器生成不同类别的样本
            for i in g_optimizers:
                i.step()

            # 每10轮保存一次生成的样本统计信息（不使用图像可视化，因为这是表格数据）
            if e % 10 == 0:
                with torch.no_grad():
                    # 设置所有生成器为评估模式
                    for i in self.generators:
                        i.eval()
                    # 为每个标签生成10个样本并拼接，使用实际的标签数量
                    generated_samples = torch.cat([self.generate_samples(i, 10) for i in range(datasets.label_num)])
                    # 恢复生成器为训练模式
                    for i in self.generators:
                        i.train()
                # 创建一个简单的散点图来可视化前两个特征（如果特征数量足够）
                if generated_samples.shape[1] >= 2:
                    plt.figure(figsize=(10, 8))
                    # 按类别绘制散点图，使用实际的标签数量
                    for i in range(datasets.label_num):
                        class_samples = generated_samples[i*10:(i+1)*10]
                        plt.scatter(class_samples[:, 0], class_samples[:, 1], label=f'Class {i}', alpha=0.6)
                    plt.title(f'TMG-GAN Generated Samples (Epoch {e})')
                    plt.xlabel('Feature 0')
                    plt.ylabel('Feature 1')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    # 保存图表
                    plt.savefig(config.path_config.gan_outs / f'tmg_{e}.jpg')
                    # 关闭图表以释放内存
                    plt.close()
                else:
                    # 如果特征数量不足，只记录基本统计信息
                    print(f"\nEpoch {e}: Generated samples shape: {generated_samples.shape}")

        # 打印换行符，结束训练进度显示
        print('')
        # 设置CD模型为评估模式
        self.cd.eval()
        # 设置所有生成器为评估模式
        for i in self.generators:
            i.eval()

    def _divide_samples(self, dataset: datasets.TrDataset) -> None:
        """将数据集按标签划分到字典中
        
        Args:
            dataset: 训练数据集
        """
        # 遍历数据集中的每个样本和标签
        for sample, label in dataset:
            # 将标签转换为Python整数
            label = label.item()
            # 如果当前标签不在字典中，创建新条目
            if label not in self.samples.keys():
                self.samples[label] = sample.unsqueeze(0)  # 添加批次维度
            else:
                # 将新样本添加到现有标签的样本集中
                self.samples[label] = torch.cat([self.samples[label], sample.unsqueeze(0)])

    def _get_target_samples(self, label: int, num: int) -> torch.Tensor:
        """随机选择指定数量的目标标签样本
        
        Args:
            label: 目标标签
            num: 需要的样本数量
            
        Returns:
            选定的样本张量
        """
        return torch.stack(
            random.choices(
                self.samples[label],  # 从指定标签的样本中选择
                k=num,  # 选择num个样本
            )
        )

    def generate_samples(self, target_label: int, num: int):
        """生成指定标签和数量的样本
        
        Args:
            target_label: 目标类别标签
            num: 需要生成的样本数量
            
        Returns:
            生成的样本，已移至CPU并从计算图中分离
        """
        # 使用对应标签的生成器生成样本，移至CPU并从计算图中分离
        return self.generators[target_label].generate_samples(num).cpu().detach()

    def generate_qualified_samples(self, target_label: int, num: int):
        """生成经过分类器验证的合格样本
        
        Args:
            target_label: 目标类别标签
            num: 需要生成的合格样本数量
            
        Returns:
            生成的合格样本
        """
        # 存储合格样本的列表
        result = []
        # 重试耐心值，防止无限循环
        patience = 10
        # 当合格样本数量不足时继续生成
        while len(result) < num:
            # 生成一个样本
            sample = self.generators[target_label].generate_samples(1)
            # 使用CD模型预测样本的标签
            label = torch.argmax(self.cd(sample)[1])
            # 如果预测标签正确或耐心值为0，则接受该样本
            if label == target_label or patience == 0:
                result.append(sample.cpu().detach())
                # 重置耐心值
                patience = 10
            else:
                # 减少耐心值
                patience -= 1
        # 拼接所有合格样本并返回
        return torch.cat(result)
