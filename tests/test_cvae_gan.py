# 首先设置环境变量，确保在导入sklearn之前生效，解决joblib多进程警告
import os
import multiprocessing
# 设置LOKY_MAX_CPU_COUNT环境变量，限制joblib使用的CPU数量为系统实际CPU数量
os.environ['LOKY_MAX_CPU_COUNT'] = str(multiprocessing.cpu_count())

# 导入上下文模块，用于设置项目路径等环境
import context

# 导入PyTorch深度学习框架
import torch

# 导入项目的src模块，包含所有核心功能
import src

# 导入seaborn数据可视化库，用于绘制美观的统计图形
import seaborn as sns
# 导入matplotlib.pyplot，用于图形绘制和显示
import matplotlib.pyplot as plt
# 从sklearn.manifold导入TSNE算法，用于高维数据降维可视化
from sklearn.manifold import TSNE

# 主程序入口，确保该代码块只在直接运行脚本时执行
if __name__ == '__main__':
    # 注释掉的代码，用于开启测试模式，限制数据集大小以便快速测试
    # src.utils.turn_on_test_mode()
    
    # 调用src.utils模块的prepare_datasets函数，准备并加载数据集
    src.utils.prepare_datasets()
    
    # 设置随机种子，确保实验结果可重复
    src.utils.set_random_state()
    
    # 创建CVAEGAN模型实例
    gan = src.CVAEGAN()
    
    # 在训练数据集上训练CVAEGAN模型
    gan.fit(src.datasets.TrDataset())
    
    # 注释：直接访问全局数据集而不是通过TrDataset的samples/labels属性
    # 获取训练样本并移至CPU（如果当前在GPU上）
    x = src.datasets.tr_samples.cpu()
    # 获取训练标签并移至CPU（如果当前在GPU上）
    y = src.datasets.tr_labels.cpu()

    # 遍历所有类别（根据数据集的类别数量）
    for i in range(src.datasets.label_num):
        # 为每个类别生成与该类别原始样本数量相同的新样本，并与原始样本合并
        # len(gan.samples[i])获取类别i的原始样本数量
        x = torch.cat([x, gan.generate_samples(i, len(gan.samples[i]))])
        
        # 为生成的样本创建标签，标签值为原始类别值+0.1
        # 这样可以在可视化中区分原始样本和生成样本
        # 例如：原始样本标签为0，生成样本标签为0.1
        y = torch.cat([y, torch.full([len(gan.samples[i])], i + 0.1)])

    # 使用t-SNE算法对样本进行降维，将高维特征转换为2维以便可视化
    embedded_x = TSNE(
        learning_rate='auto',  # 自动选择合适的学习率
        init='random',         # 使用随机初始化
        random_state=src.config.seed,  # 设置随机种子，确保结果可重复
    ).fit_transform(x)  # 对样本进行拟合和转换，得到降维后的结果
    
    # 使用seaborn创建散点图
    sns.scatterplot(
        x=embedded_x[:, 0],  # x轴使用t-SNE降维后的第一个维度
        y=embedded_x[:, 1],  # y轴使用t-SNE降维后的第二个维度
        hue=y,               # 使用标签作为颜色区分，不同标签显示不同颜色
        palette="deep",       # 使用deep调色板，颜色丰富且区分度高
        alpha=1,             # 设置点的透明度为1（完全不透明）
    )
    
    # 保存散点图到指定路径
    plt.savefig('tests/Visualization_results/cvae-gan.jpg')
    
    # 在窗口中显示散点图
    plt.show()
