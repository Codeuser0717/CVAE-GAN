import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import torch
import numpy as np
from src import Classifier, datasets, utils
from src.sngan import SNGAN
import time

dataset = 'CAN_HCRL_OTIDS'
# datasets='car_hacking'

if __name__ == '__main__':
    start_time = time.time()
    utils.set_random_state()
    lens = (len(datasets.tr_samples), len(datasets.te_samples))
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

    # 初始化SNGAN模型
    utils.set_random_state()
    sngan = SNGAN()
    
    # 训练SNGAN模型
    sngan.fit(datasets.TrDataset())
    
    # 绘制并保存损失历史曲线
    print("\n生成损失曲线...")
    sngan.plot_loss_history()
    
    # 打印原始数据集类别分布
    print("原始数据集类别分布:")
    original_class_counts = {}
    for i in sngan.samples.keys():
        original_class_counts[i] = len(sngan.samples[i])
        print(f"类别 {i}: {original_class_counts[i]} 个样本")
    
    # 计算每个类别的最大样本数
    max_cnt = max([len(sngan.samples[i]) for i in sngan.samples.keys()])

    total_generated = 0
    generation_stats = {}
    
    for i in sngan.samples.keys():
        current_count = len(sngan.samples[i])
        cnt_generated = max_cnt - current_count
        generation_stats[i] = {'target': cnt_generated, 'actual': 0}
        
        if cnt_generated > 0:
            print(f"为类别 {i} 生成 {cnt_generated} 个样本...")
            
            # 调用SNGAN生成样本
            generated_samples = sngan.generate_qualified_samples(i, cnt_generated)
            actual_generated = len(generated_samples)
            generation_stats[i]['actual'] = actual_generated
            
            print(f"类别 {i} 生成完成: 目标 {cnt_generated}, 实际生成 {actual_generated}")
            
            if actual_generated > 0:
                generated_labels = torch.full([actual_generated], i)
                datasets.tr_samples = torch.cat([datasets.tr_samples, generated_samples])
                datasets.tr_labels = torch.cat([datasets.tr_labels, generated_labels])
                total_generated += actual_generated
        else:
            print(f"类别 {i} 已达到最大样本数 {current_count}，无需生成")
    # 计算增强后的类别分布
    enhanced_class_counts = {}
    # 将标签转换为numpy数组以便统计
    labels_np = datasets.tr_labels.numpy()
    for i in range(datasets.label_num):
        count = np.sum(labels_np == i)
        enhanced_class_counts[i] = count
        print(f"类别 {i}: {count} 个样本")

    # 确保特征和标签数量一致
    assert len(datasets.tr_samples) == len(datasets.tr_labels), f"特征和标签数量不匹配: {len(datasets.tr_samples)} vs {len(datasets.tr_labels)}"

    # 重新创建训练数据集对象
    train_dataset = datasets.TrDataset()
    print(f"\n新数据集对象大小: {len(train_dataset)}")

    # 保存增强后的数据集
    with open('data_sngan.pkl', 'wb') as f:
        pickle.dump(
            (
                datasets.tr_samples.numpy(),
                datasets.tr_labels.numpy(),
                datasets.te_samples.numpy(),
                datasets.te_labels.numpy(),
            ),
            f,
        )

    # 使用SNGAN的分类器
    utils.set_random_state()
    clf = Classifier('SNGAN')
    clf.model = sngan.classifier  # 使用SNGAN训练好的分类器
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
    
    # 额外功能：分析谱归一化的效果
    print("\n分析谱归一化效果...")
    with torch.no_grad():
        sngan.generator.eval()
        sngan.discriminator.eval()
        
        # 检查判别器权重谱范数
        for name, module in sngan.discriminator.named_modules():
            if hasattr(module, 'weight_orig'):
                # 计算谱范数
                weight = module.weight_orig
                u = module.weight_u
                v = module.weight_v
                
                # 估计谱范数
                sigma = torch.dot(u.flatten(), torch.matmul(weight, v).flatten())
                print(f"判别器层 {name} 的谱范数估计: {sigma.item():.4f}")
        
        # 为每个类别生成一些样本
        for label in range(datasets.label_num):
            # 生成5个样本
            generated = sngan.generate_samples(label, 5)
            
            if len(generated) > 0:
                # 计算生成样本的统计信息
                mean_val = torch.mean(generated).item()
                std_val = torch.std(generated).item()
                print(f"类别 {label} 的生成样本统计 - 均值: {mean_val:.6f}, 标准差: {std_val:.6f}")