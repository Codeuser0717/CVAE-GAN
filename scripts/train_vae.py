import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pickle
import torch
import numpy as np
from src import Classifier, datasets, utils
from src.vae import VAE
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

    # 初始化VAE模型
    utils.set_random_state()
    vae = VAE()
    
    # 训练VAE模型
    print("开始训练VAE模型...")
    vae.fit(datasets.TrDataset())
    
    # 绘制并保存损失历史曲线
    print("\n生成损失曲线...")
    vae.plot_loss_history()
    
    # 打印原始数据集大小
    print(f"原始训练集大小: {len(datasets.tr_samples)}")
    
    # 由于VAE无法按类别生成样本，我们生成总体的样本，然后使用分类器筛选
    print("\n使用VAE生成样本并平衡数据集...")
    
    # 先训练一个分类器来评估生成样本的类别
    # 这里我们使用VAE内置的分类器，但需要重新训练一下
    print("训练分类器用于评估生成样本...")
    
    # 计算每个类别的样本数
    class_counts = {}
    for i in range(datasets.label_num):
        class_counts[i] = torch.sum(datasets.tr_labels == i).item()
        print(f"类别 {i}: {class_counts[i]} 个样本")
    
    # 找出最大样本数
    max_cnt = max(class_counts.values())
    print(f"最大样本类别数量: {max_cnt}")
    
    # 为少数类别生成样本以平衡数据集
    print("\n开始生成平衡样本...")
    total_generated = 0
    generation_stats = {}
    
    for i in range(datasets.label_num):
        current_count = class_counts[i]
        cnt_generated = max_cnt - current_count
        generation_stats[i] = {'target': cnt_generated, 'actual': 0}
        
        if cnt_generated > 0:
            print(f"尝试为类别 {i} 生成 {cnt_generated} 个样本...")
            
            # 调用VAE生成样本，并用分类器筛选
            generated_samples = vae.generate_qualified_samples(i, cnt_generated)
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

    # 重新创建训练数据集对象
    train_dataset = datasets.TrDataset()
    print(f"\n新数据集对象大小: {len(train_dataset)}")

    # 保存增强后的数据集
    with open('data_vae.pkl', 'wb') as f:
        pickle.dump(
            (
                datasets.tr_samples.numpy(),
                datasets.tr_labels.numpy(),
                datasets.te_samples.numpy(),
                datasets.te_labels.numpy(),
            ),
            f,
        )

    # 使用VAE的分类器
    utils.set_random_state()
    clf = Classifier('VAE')
    clf.model = vae.classifier  # 使用VAE训练好的分类器
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
    print("\n保存生成样本用于分析...")
    with torch.no_grad():
        # 生成一些样本
        generated = vae.generate_samples(100)
        
        if len(generated) > 0:
            # 计算生成样本的统计信息
            mean_val = torch.mean(generated).item()
            std_val = torch.std(generated).item()
            print(f"生成样本统计 - 均值: {mean_val:.6f}, 标准差: {std_val:.6f}")
            
            # 使用分类器分析生成样本的类别分布
            classifier_output = vae.classifier(generated.to(config.device))
            probs = torch.softmax(classifier_output, dim=1)
            _, preds = torch.max(probs, dim=1)
            
            print("生成样本的类别分布:")
            for i in range(datasets.label_num):
                count = torch.sum(preds == i).item()
                print(f"类别 {i}: {count} 个样本")
            
            # 重构一些真实样本
            real_samples, real_labels = vae._get_random_samples_with_labels(10)
            reconstructed = vae.reconstruct_samples(real_samples)
            
            # 计算重构误差
            recon_error = torch.mean(torch.abs(real_samples - reconstructed)).item()
            print(f"样本重构误差: {recon_error:.6f}")