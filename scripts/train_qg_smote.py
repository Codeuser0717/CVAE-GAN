import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import torch
import numpy as np
from src import Classifier, datasets, utils
from src.qg_smote import QG_SMOTE
import time

dataset = 'CAN_HCRL_OTIDS'
# datasets='car_hacking'

if __name__ == '__main__':
    start_time = time.time()
    utils.set_random_state()
    print("不使用PCA，仅进行数据归一化...")
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

    # 初始化QG-SMOTE模型
    utils.set_random_state()
    qg_smote = QG_SMOTE()
    
    # 训练QG-SMOTE模型
    print("开始训练QG-SMOTE模型...")
    qg_smote.fit(datasets.TrDataset())
    
    # 绘制并保存损失历史曲线
    print("\n生成损失曲线...")
    qg_smote.plot_loss_history()
    
    # 打印原始数据集类别分布
    print("原始数据集类别分布:")
    original_class_counts = {}
    for i in qg_smote.samples.keys():
        original_class_counts[i] = len(qg_smote.samples[i])
        print(f"类别 {i}: {original_class_counts[i]} 个样本")
    
    # 计算每个类别的最大样本数
    max_cnt = max([len(qg_smote.samples[i]) for i in qg_smote.samples.keys()])
    print(f"最大样本类别数量: {max_cnt}")
    
    # 为少数类别生成样本以平衡数据集
    print("\n开始生成平衡样本...")
    total_generated = 0
    generation_stats = {}
    
    for i in qg_smote.samples.keys():
        current_count = len(qg_smote.samples[i])
        cnt_generated = max_cnt - current_count
        generation_stats[i] = {'target': cnt_generated, 'actual': 0}
        
        if cnt_generated > 0:
            print(f"为类别 {i} 生成 {cnt_generated} 个样本...")
            
            # 使用分位数方法生成样本（QG-SMOTE的核心优势）
            generated_samples = qg_smote.generate_qualified_samples(i, cnt_generated, method='quantile')
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
    with open('data_qg_smote.pkl', 'wb') as f:
        pickle.dump(
            (
                datasets.tr_samples.numpy(),
                datasets.tr_labels.numpy(),
                datasets.te_samples.numpy(),
                datasets.te_labels.numpy(),
            ),
            f,
        )

    # 使用QG-SMOTE的分类器
    utils.set_random_state()
    clf = Classifier('QG_SMOTE')
    clf.model = qg_smote.classifier  # 使用QG-SMOTE训练好的分类器

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
    
    # 额外功能：分析分位数分布
    print("\n分析各类别的分位数分布...")
    for label in range(datasets.label_num):
        if label in qg_smote.samples:
            print(f"\n分析类别 {label} 的分位数分布:")
            stats = qg_smote.analyze_quantiles(label, num_samples=50)
            
            print(f"分位数均值形状: {stats['mean'].shape}")
            print(f"分位数标准差形状: {stats['std'].shape}")
            
            # 打印前几个特征的分位数统计
            for feat_idx in range(min(3, datasets.feature_num)):
                print(f"特征 {feat_idx} 的分位数统计:")
                for q_idx in range(config.gan_config.qg_smote_config['num_quantiles']):
                    print(f"  分位数 {q_idx+1}: 均值={stats['mean'][feat_idx, q_idx]:.4f}, "
                          f"标准差={stats['std'][feat_idx, q_idx]:.4f}")
    
    # 比较不同生成方法的效果
    print("\n比较不同生成方法的效果...")
    with torch.no_grad():
        for label in range(min(3, datasets.label_num)):
            if label in qg_smote.samples:
                # 使用分位数方法生成
                quantile_samples = qg_smote.generate_samples(label, 10, method='quantile')
                # 使用先验方法生成
                prior_samples = qg_smote.generate_samples(label, 10, method='prior')
                
                # 计算多样性（样本间的平均距离）
                quantile_diversity = torch.mean(torch.cdist(quantile_samples, quantile_samples)).item()
                prior_diversity = torch.mean(torch.cdist(prior_samples, prior_samples)).item()
                
                print(f"类别 {label}: 分位数方法多样性={quantile_diversity:.4f}, "
                      f"先验方法多样性={prior_diversity:.4f}")