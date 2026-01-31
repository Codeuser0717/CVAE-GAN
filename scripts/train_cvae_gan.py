import sys
import os
import multiprocessing
os.environ['LOKY_MAX_CPU_COUNT'] = str(multiprocessing.cpu_count())
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import torch
import numpy as np
from src import models, Classifier, datasets, utils
from src.cvae_gan import CVAEGAN
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

    # 初始化CVAE-GAN模型
    utils.set_random_state()
    cvae_gan = CVAEGAN()
    
    # 训练CVAE-GAN模型
    print("开始训练CVAE-GAN模型...")
    cvae_gan.fit(datasets.TrDataset())
    
    # 绘制并保存损失历史曲线
    print("\n生成损失曲线...")
    cvae_gan.plot_loss_history()
    
    # 打印原始数据集类别分布
    print("原始数据集类别分布:")
    original_class_counts = {}
    for i in cvae_gan.samples.keys():
        original_class_counts[i] = len(cvae_gan.samples[i])
        print(f"类别 {i}: {original_class_counts[i]} 个样本")
    
    # 计算每个类别的最大样本数
    max_cnt = max([len(cvae_gan.samples[i]) for i in cvae_gan.samples.keys()])
    print(f"最大样本类别数量: {max_cnt}")
    
    # 为少数类别生成样本以平衡数据集
    print("\n开始生成平衡样本...")
    total_generated = 0
    total_attempts = 0
    generation_stats = {}
    
    for i in cvae_gan.samples.keys():
        current_count = len(cvae_gan.samples[i])
        cnt_generated = max_cnt - current_count
        generation_stats[i] = {'target': cnt_generated, 'actual': 0}
        
        if cnt_generated > 0:
            print(f"为类别 {i} 生成 {cnt_generated} 个样本...")
            
            # 调用CVAEGAN生成样本
            generated_samples = cvae_gan.generate_qualified_samples(i, cnt_generated)
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
    
    # 打印生成统计摘要
    print("\n生成统计摘要:")
    for i in generation_stats:
        if generation_stats[i]['target'] > 0:
            success_rate = (generation_stats[i]['actual'] / generation_stats[i]['target']) * 100 if generation_stats[i]['target'] > 0 else 0
            print(f"类别 {i}: 目标 {generation_stats[i]['target']}, 实际生成 {generation_stats[i]['actual']}, 成功率: {success_rate:.2f}%")
    
    # 打印增强前后的数据集大小对比
    original_size = len(datasets.tr_samples) - total_generated
    print(f"\n数据集增强统计:")
    print(f"增强前训练集大小: {original_size}")

    print(f"总共生成 {total_generated} 个样本")
    print(f"增强后训练集大小: {len(datasets.tr_samples)}")
    print(f"增强比例: {(total_generated / original_size) * 100:.2f}%")

    # 计算增强后的类别分布
    print("\n增强后数据集类别分布:")
    enhanced_class_counts = {}
    # 将标签转换为numpy数组以便统计
    labels_np = datasets.tr_labels.numpy()
    for i in range(datasets.label_num):
        count = np.sum(labels_np == i)
        enhanced_class_counts[i] = count
        print(f"类别 {i}: {count} 个样本")

    # 确保特征和标签数量一致
    assert len(datasets.tr_samples) == len(datasets.tr_labels), f"特征和标签数量不匹配: {len(datasets.tr_samples)} vs {len(datasets.tr_labels)}"

    # 重新创建训练数据集对象（关键修改！）
    train_dataset = datasets.TrDataset()
    print(f"\n新数据集对象大小: {len(train_dataset)}")

    # 保存增强后的数据集
    with open('data_cvae_gan.pkl', 'wb') as f:
        pickle.dump(
            (
                datasets.tr_samples.numpy(),
                datasets.tr_labels.numpy(),
                datasets.te_samples.numpy(),
                datasets.te_labels.numpy(),
            ),
            f,
        )

    # 使用CVAE-GAN的判别器-分类器作为分类器
    utils.set_random_state()
    clf = Classifier('CVAE_GAN')
    clf.model = cvae_gan.classifier  # 使用CVAE-GAN训练好的判别器-分类器

    # 在增强后的数据集上微调分类器
    print("在增强数据集上微调分类器...")
    # 验证数据集大小
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
        # 为每个类别生成一些样本
        for label in range(datasets.label_num):
            # 生成5个样本
            generated = cvae_gan.generate_samples(label, 5)
            
            if len(generated) > 0:
                # 计算生成样本的统计信息
                mean_val = torch.mean(generated).item()
                std_val = torch.std(generated).item()
                print(f"类别 {label} 的生成样本统计 - 均值: {mean_val:.6f}, 标准差: {std_val:.6f}")