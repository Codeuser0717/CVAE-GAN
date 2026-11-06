import context

import os
import multiprocessing
# 设置LOKY_MAX_CPU_COUNT环境变量以解决joblib警告
os.environ['LOKY_MAX_CPU_COUNT'] = str(multiprocessing.cpu_count())

import pickle

import torch

import src
from src import Classifier, datasets, utils
import time
from src.tmg_gan import TMGGAN

dataset = 'CAN_HCRL_OTIDS'
# dataset = 'NSL-KDD'

if __name__ == '__main__':
    # 新增：记录开始时间
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

    src.utils.set_random_state()
    tmg_gan = src.TMGGAN()
    tmg_gan.fit(src.datasets.TrDataset())
    # count the max number of samples
    max_cnt = max([len(tmg_gan.samples[i]) for i in tmg_gan.samples.keys()])
    # generate samples
    for i in tmg_gan.samples.keys():
        cnt_generated = max_cnt - len(tmg_gan.samples[i])
        if cnt_generated > 0:
            generated_samples = tmg_gan.generate_qualified_samples(i, cnt_generated)
            generated_labels = torch.full([cnt_generated], i)
            datasets.tr_samples = torch.cat([datasets.tr_samples, generated_samples])
            datasets.tr_labels = torch.cat([datasets.tr_labels, generated_labels])

    with open('data.pkl', 'wb') as f:
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
    clf = Classifier('TMG_GAN')
    clf.model = tmg_gan.cd
    clf.fit(datasets.TrDataset())
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
# 新增：记录结束时间并计算总执行时间
    end_time = time.time()
    total_time = end_time - start_time
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"\n总执行时间: {int(hours)}h {int(minutes)}m {seconds:.2f}s")