import context
import os
import multiprocessing
os.environ['LOKY_MAX_CPU_COUNT'] = str(multiprocessing.cpu_count())
import pickle
import torch
import src
from src import Classifier, datasets, utils
import time
from src.tmg_gan import TMGGAN

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