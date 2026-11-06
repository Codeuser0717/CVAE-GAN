import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
from pathlib import Path
# 设置路径（使用绝对路径，不依赖src模块）
BASE_DIR = Path(__file__).absolute().parent.parent
DATA_DIR = BASE_DIR / 'data'
DATASET_DIR = DATA_DIR / 'datasets' / 'CAN_HCRL_OTIDS'
INPUT_FILE = DATASET_DIR / 'CAN data.csv'
# 确保输出目录存在
os.makedirs(DATASET_DIR, exist_ok=True)
print("开始处理CAN_HCRL_OTIDS数据集...")
# 读取CSV文件
print(f"读取数据文件: {INPUT_FILE}")
df = pd.read_csv(INPUT_FILE)
print(f"数据集形状: {df.shape}")
print(f"数据集列名: {list(df.columns)}")
# 假设最后一列是标签列
# 请根据实际情况调整标签列名
label_column = df.columns[-1]
print(f"假设标签列为: {label_column}")
# 分离特征和标签
X = df.drop(label_column, axis=1)
y = df[label_column]
# 自定义标签顺序（替换为您想要的顺序）
custom_label_order = ['R', 'RPM', 'gear', 'DoS', 'Fuzzy']  # 这里是自定义的顺序，您可以根据需要修改

# 检查自定义顺序是否包含所有原始类别
original_classes = list(set(y))
if not all(cls in custom_label_order for cls in original_classes):
    print("警告: 自定义标签顺序未包含所有原始类别！")
    # 自动添加缺失的类别
    for cls in original_classes:
        if cls not in custom_label_order:
            custom_label_order.append(cls)
    print(f"已自动调整顺序: {custom_label_order}")

# 使用自定义顺序创建映射字典
label_to_index = {label: i for i, label in enumerate(custom_label_order)}

# 手动编码标签
y_encoded = np.array([label_to_index[label] for label in y])

# 获取类别数量（移到前面定义，供后续使用）
num_classes = len(custom_label_order)
# 设置随机种子（移到前面定义，供后续使用）
random_state = 42
print(f"自定义标签类别顺序: {custom_label_order}")
print(f"编码后的标签值: {list(np.unique(y_encoded))}")
print(f"类别数量: {num_classes}")
# 显示原始数据集中每类样本数量
print("\n原始数据集中每类样本数量:")
class_counts = np.bincount(y_encoded)
for i, (class_name, count) in enumerate(zip(custom_label_order, class_counts)):
    print(f"类别 {i} ({class_name}): {count} 个样本")
# 设置每类的采样比例
default_sample_ratio = 1.0  # 默认采样比例
# 未在字典中指定的类别将使用默认采样比例
sample_ratios = {
    #can
    0: 0.01, 
    1: 0.1,  
    2: 0.1, 
    3: 0.1,
    4: 0.1, 
}

print(f"\n设置各类别采样比例...")

# 创建DataFrame以便按类别采样
df_combined = pd.DataFrame(X)
df_combined['label'] = y_encoded

# 按类别采样
sampled_data = []
sampled_labels = []

total_original = 0
total_sampled = 0

for class_idx in range(num_classes):
    class_data = df_combined[df_combined['label'] == class_idx]
    class_count = len(class_data)
    
    # 获取该类别的采样比例，如果没有指定则使用默认值
    class_ratio = sample_ratios.get(class_idx, default_sample_ratio)
    sample_size = max(1, int(class_count * class_ratio))  # 确保至少采样1个样本
    
    # 随机采样
    if class_count > 0:
        sampled_class_data = class_data.sample(n=sample_size, random_state=random_state)
        sampled_data.append(sampled_class_data.drop('label', axis=1))
        sampled_labels.extend(sampled_class_data['label'].values)
    
    print(f"类别 {class_idx} ({custom_label_order[class_idx]}): 原始 {class_count} 个，采样比例 {class_ratio*100:.1f}%，采样 {sample_size} 个")
    total_original += class_count
    total_sampled += sample_size if class_count > 0 else 0

# 合并采样后的数据
X_sampled = pd.concat(sampled_data).values
y_sampled = np.array(sampled_labels)

print(f"\n采样后总样本数: {len(X_sampled)}")
print(f"采样前总样本数: {len(X)}")
print(f"采样比例: {len(X_sampled) / len(X) * 100:.2f}%")

# 分割训练集和测试集
test_size = 0.2  # 20%用于测试
random_state = 42  # 确保结果可重复

print("\n分割训练集和测试集...")
X_train, X_test, y_train, y_test = train_test_split(
    X_sampled, y_sampled, test_size=test_size, random_state=random_state, stratify=y_sampled
)

print(f"训练集大小: {X_train.shape}")
print(f"测试集大小: {X_test.shape}")

# 显示训练集中每类样本数量
print("\n训练集中每类样本数量:")
train_class_counts = np.bincount(y_train)
for i, (class_name, count) in enumerate(zip(custom_label_order, train_class_counts)):
    print(f"类别 {i} ({class_name}): {count} 个样本")

# 显示测试集中每类样本数量
print("\n测试集中每类样本数量:")
test_class_counts = np.bincount(y_test)
for i, (class_name, count) in enumerate(zip(custom_label_order, test_class_counts)):
    print(f"类别 {i} ({class_name}): {count} 个样本")

# 将标签转换为one-hot编码格式（与现有代码兼容）
def to_one_hot(labels, num_classes):
    """将标签转换为one-hot编码格式"""
    one_hot = np.zeros((len(labels), num_classes))
    for i, label in enumerate(labels):
        one_hot[i, label] = 1
    return one_hot

# 类别数量已在前面定义

# 转换为one-hot编码
y_train_one_hot = to_one_hot(y_train, num_classes)
y_test_one_hot = to_one_hot(y_test, num_classes)

# 保存文件
print("保存处理后的数据集...")

# 保存特征文件 - 将numpy数组转换为DataFrame后保存
pd.DataFrame(X_train).to_csv(DATASET_DIR / 'x_train.csv', index=False)
pd.DataFrame(X_test).to_csv(DATASET_DIR / 'x_test.csv', index=False)

# 保存标签文件
pd.DataFrame(y_train_one_hot).to_csv(DATASET_DIR / 'y_train.csv', index=False)
pd.DataFrame(y_test_one_hot).to_csv(DATASET_DIR / 'y_test.csv', index=False)

print("数据集预处理完成！")
print(f"训练特征文件: {DATASET_DIR / 'x_train.csv'}")
print(f"训练标签文件: {DATASET_DIR / 'y_train.csv'}")
print(f"测试特征文件: {DATASET_DIR / 'x_test.csv'}")
print(f"测试标签文件: {DATASET_DIR / 'y_test.csv'}")