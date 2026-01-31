import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import multiprocessing
os.environ['LOKY_MAX_CPU_COUNT'] = str(multiprocessing.cpu_count())
import pickle
import torch
import numpy as np
from src import Classifier, datasets, utils, VAEGAN
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
    utils.set_random_state()
    clf = Classifier('VAE_GAN')
    clf.fit(train_dataset)

    # 清理GPU内存
    torch.cuda.empty_cache()

    test_dataset = datasets.TeDataset()
    clf.test(test_dataset)
    print(clf.confusion_matrix)
    clf.print_metrics(4)  
    # 绘制多分类ROC曲线
    clf.plot_roc_curve(test_dataset, is_binary=False)

    # 二分类测试（使用同一个测试数据集实例）
    print("二分类测试...")
    clf.binary_test(test_dataset)
    print(clf.confusion_matrix)
    clf.print_metrics(4)  
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