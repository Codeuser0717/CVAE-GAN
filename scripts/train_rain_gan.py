import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import torch
import numpy as np
from src import Classifier, datasets, utils
from src.rain_gan import RAIN_GAN
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

    # 初始化RAIN-GAN模型
    utils.set_random_state()
    rain_gan = RAIN_GAN()
    
    # 训练RAIN-GAN模型
    print("开始训练RAIN-GAN模型...")
    rain_gan.fit(datasets.TrDataset())
    
    # 绘制并保存损失历史曲线
    print("\n生成损失曲线...")
    rain_gan.plot_loss_history()
    
    total_generated = 0
    generation_stats = {}
    
    for i in rain_gan.samples.keys():
        current_count = len(rain_gan.samples[i])
        cnt_generated = max_cnt - current_count
        generation_stats[i] = {'target': cnt_generated, 'actual': 0}
        
        if cnt_generated > 0:
            print(f"为类别 {i} 生成 {cnt_generated} 个样本...")
            
            # 调用RAIN-GAN生成样本
            generated_samples = rain_gan.generate_qualified_samples(i, cnt_generated)
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
    
    # 确保特征和标签数量一致
    assert len(datasets.tr_samples) == len(datasets.tr_labels), f"特征和标签数量不匹配: {len(datasets.tr_samples)} vs {len(datasets.tr_labels)}"

    # 重新创建训练数据集对象
    train_dataset = datasets.TrDataset()
    print(f"\n新数据集对象大小: {len(train_dataset)}")

    # 保存增强后的数据集
    with open('data_rain_gan.pkl', 'wb') as f:
        pickle.dump(
            (
                datasets.tr_samples.numpy(),
                datasets.tr_labels.numpy(),
                datasets.te_samples.numpy(),
                datasets.te_labels.numpy(),
            ),
            f,
        )

    # 使用RAIN-GAN的分类器
    utils.set_random_state()
    clf = Classifier('RAIN_GAN')
    clf.model = rain_gan.classifier  # 使用RAIN-GAN训练好的分类器
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
    
    # 可视化注意力机制
    print("\n分析注意力机制...")
    with torch.no_grad():
        # 为每个类别分析一些样本
        for label in range(min(3, datasets.label_num)):
            if label in rain_gan.samples:
                # 获取一些真实样本
                real_samples = rain_gan._get_target_samples(label, 5)
                real_labels = torch.full([5], label)
                
                # 可视化注意力
                attention_info = rain_gan.visualize_attention(real_samples, real_labels)
                
                print(f"类别 {label} 的注意力分析:")
                
                if attention_info['encoder_attention'] is not None:
                    encoder_attn_mean = np.mean(attention_info['encoder_attention'])
                    encoder_attn_std = np.std(attention_info['encoder_attention'])
                    print(f"  编码器注意力 - 均值: {encoder_attn_mean:.4f}, 标准差: {encoder_attn_std:.4f}")
                
                if attention_info['classifier_attention'] is not None:
                    classifier_attn_mean = np.mean(attention_info['classifier_attention'])
                    classifier_attn_std = np.std(attention_info['classifier_attention'])
                    print(f"  分类器注意力 - 均值: {classifier_attn_mean:.4f}, 标准差: {classifier_attn_std:.4f}")
        
        # 分析生成样本的质量
        print("\n分析生成样本的质量...")
        for label in range(min(2, datasets.label_num)):
            if label in rain_gan.samples:
                # 生成一些样本
                generated = rain_gan.generate_samples(label, 10)
                
                if len(generated) > 0:
                    # 计算生成样本的统计信息
                    mean_val = torch.mean(generated).item()
                    std_val = torch.std(generated).item()
                    print(f"类别 {label} 的生成样本统计 - 均值: {mean_val:.6f}, 标准差: {std_val:.6f}")
                    
                    # 重构一些真实样本
                    real_samples = rain_gan._get_target_samples(label, 5)
                    real_labels = torch.full([5], label)
                    reconstructed = rain_gan.reconstruct_samples(real_samples, real_labels)
                    
                    # 计算重构误差
                    recon_error = torch.mean(torch.abs(real_samples - reconstructed)).item()
                    print(f"类别 {label} 的重构误差: {recon_error:.6f}")