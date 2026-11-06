import sys
import os
# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import multiprocessing
os.environ['LOKY_MAX_CPU_COUNT'] = str(multiprocessing.cpu_count())

import pickle
import torch
import numpy as np
from src import Classifier, datasets, utils, VAEGAN
import time

if __name__ == '__main__':
    # 记录开始时间
    start_time = time.time()
    utils.set_random_state()
    
    # 不使用PCA降维，只进行归一化处理
    print("不使用PCA，仅进行数据归一化...")
    # 保存原始样本数量，用于后续分割
    lens = (len(datasets.tr_samples), len(datasets.te_samples))
    # 合并训练集和测试集以进行统一的归一化处理
    samples = torch.cat(
        [
            datasets.tr_samples,
            datasets.te_samples,
        ]
    )
    # 转换为numpy数组进行sklearn处理
    samples_np = samples.numpy()
    
    # 导入必要的库
    from sklearn.preprocessing import minmax_scale
    
    # 直接对原始数据进行归一化处理，不进行降维
    normalized_samples = minmax_scale(samples_np)
    # 转换回PyTorch张量
    samples = torch.from_numpy(normalized_samples).float()
    # 确保数据范围非负
    samples = (samples - samples.min())
    
    # 重新分割训练集和测试集
    datasets.tr_samples, datasets.te_samples = torch.split(samples, lens)
    # 更新数据集的特征数量
    utils.set_dataset_values()
    print(f"使用原始特征数量: {datasets.feature_num}")

    # 初始化VAE-GAN模型
    utils.set_random_state()
    vae_gan = VAEGAN()
    
    # 训练VAE-GAN模型
    print("开始训练VAE-GAN模型...")
    vae_gan.fit(datasets.TrDataset())
    
    # 绘制并保存损失历史曲线
    print("\n生成损失曲线...")
    vae_gan.plot_loss_history()
    
    # 打印原始数据集大小
    print(f"原始训练集大小: {len(datasets.tr_samples)}")
    
    # 生成一些样本以平衡数据集（如果需要）
    print("\n生成样本以增强数据集...")
    target_size = len(datasets.tr_samples) * 2  # 目标大小设为原大小的2倍
    num_to_generate = max(0, target_size - len(datasets.tr_samples))
    
    if num_to_generate > 0:
        print(f"需要生成 {num_to_generate} 个样本...")
        
        # 调用VAEGAN生成样本
        generated_samples = vae_gan.generate_samples(num_to_generate)
        
        # 生成随机标签（因为VAE-GAN是无监督的，没有特定标签）
        generated_labels = torch.randint(0, datasets.label_num, (len(generated_samples),))
        
        # 添加到训练集
        datasets.tr_samples = torch.cat([datasets.tr_samples, generated_samples])
        datasets.tr_labels = torch.cat([datasets.tr_labels, generated_labels])
        
        print(f"成功生成 {len(generated_samples)} 个样本")
        print(f"增强后训练集大小: {len(datasets.tr_samples)}")
        print(f"增强比例: {(len(generated_samples) / (len(datasets.tr_samples) - len(generated_samples))) * 100:.2f}%")
    
    # 确保特征和标签数量一致
    assert len(datasets.tr_samples) == len(datasets.tr_labels), f"特征和标签数量不匹配: {len(datasets.tr_samples)} vs {len(datasets.tr_labels)}"

    # 重新创建训练数据集对象
    train_dataset = datasets.TrDataset()
    print(f"\n新数据集对象大小: {len(train_dataset)}")

    # 保存增强后的数据集
    with open('data_vae_gan.pkl', 'wb') as f:
        pickle.dump(
            (
                datasets.tr_samples.numpy(),
                datasets.tr_labels.numpy(),
                datasets.te_samples.numpy(),
                datasets.te_labels.numpy(),
            ),
            f,
        )

    # 使用训练好的分类器（如果需要的话）
    # 注意：VAE-GAN没有内置分类器，需要单独训练一个
    utils.set_random_state()
    clf = Classifier('VAE_GAN')
    # 这里需要重新初始化分类器并训练
    
    # 在增强后的数据集上训练分类器
    print("在增强数据集上训练分类器...")
    print(f"增强后训练集样本数: {len(datasets.tr_samples)}, 标签数: {len(datasets.tr_labels)}")
    print(f"数据集实例长度: {len(train_dataset)}")
    print(f"特征维度: {datasets.feature_num}, 类别数量: {datasets.label_num}")

    clf.fit(train_dataset)

    # 清理GPU内存
    torch.cuda.empty_cache()

    # 测试分类器性能
    print("测试分类器性能...")
    # 创建测试数据集实例
    test_dataset = datasets.TeDataset()
    clf.test(test_dataset)
    print(clf.confusion_matrix)
    clf.print_metrics(4)  # 打印四位小数的评估指标
    # 绘制多分类ROC曲线
    clf.plot_roc_curve(test_dataset, is_binary=False)

    # 二分类测试（使用同一个测试数据集实例）
    print("二分类测试...")
    clf.binary_test(test_dataset)
    print(clf.confusion_matrix)
    clf.print_metrics(4)  # 打印四位小数的评估指标
    # 绘制二分类ROC曲线
    clf.plot_roc_curve(test_dataset, is_binary=True)
    
    # 记录结束时间并计算总执行时间
    end_time = time.time()
    total_time = end_time - start_time
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"\n总执行时间: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    
    # 额外功能：保存生成样本用于可视化分析
    print("\n保存生成样本用于分析...")
    with torch.no_grad():
        # 生成一些样本
        generated = vae_gan.generate_samples(100)
        
        if len(generated) > 0:
            # 计算生成样本的统计信息
            mean_val = torch.mean(generated).item()
            std_val = torch.std(generated).item()
            print(f"生成样本统计 - 均值: {mean_val:.6f}, 标准差: {std_val:.6f}")