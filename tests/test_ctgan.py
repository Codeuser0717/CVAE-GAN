# 首先设置环境变量，确保在导入sklearn之前生效
import os
import multiprocessing
os.environ['LOKY_MAX_CPU_COUNT'] = str(multiprocessing.cpu_count())

import context

import torch

import src

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

if __name__ == '__main__':
    # src.utils.turn_on_test_mode()
    src.utils.prepare_datasets()
    src.utils.set_random_state()
    gan = src.CTGAN()
    gan.fit(src.datasets.TrDataset())
    # 直接访问全局数据集而不是通过TrDataset的samples/labels属性
    x = src.datasets.tr_samples.cpu()
    y = src.datasets.tr_labels.cpu()

    for i in range(src.datasets.label_num):
        x = torch.cat([x, gan.generate_samples(i, len(gan.samples[i]), to_cpu=True)])
        y = torch.cat([y, torch.full([len(gan.samples[i])], i + 0.1)])

    embedded_x = TSNE(
        learning_rate='auto',
        init='random',
        random_state=src.config.seed,
    ).fit_transform(x)
    sns.scatterplot(
        x=embedded_x[:, 0],
        y=embedded_x[:, 1],
        hue=y,
        palette="deep",
        alpha=1,
    )
    plt.savefig('tests/Visualization_results/ctgan.jpg')
    plt.show()
