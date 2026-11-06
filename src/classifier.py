import torch
import numpy as np
from torch.nn.functional import cross_entropy
from torch.optim import Adam
from sklearn import metrics
from torch.utils.data import DataLoader

from src import config, datasets, logger, models


class Classifier:
    def __init__(self, name: str):
        self.name = f'{name}_classifier'
        self.model = models.CVAEGANClassifierModel(datasets.feature_num, datasets.label_num).to(config.device)
        self.logger = logger.Logger(name)
        self.confusion_matrix: np.ndarray = None
        self.metrics = {
            'Precision': 0.0,
            'Recall': 0.0,
            'F1': 0.0,
        }
        self.class_metrics = None  # 用于存储每个类别的详细指标

    def fit(self, dataset: datasets.TrDataset):
        self.model.train()
        self.logger.info('Started training')
        self.logger.debug(f'Using device: {config.device}')
        optimizer = Adam(
            params=self.model.parameters(),
            lr=config.classifier_config.lr,
        )
        dl = DataLoader(dataset, config.classifier_config.batch_size, shuffle=True)
        for e in range(config.classifier_config.epochs):
            for idx, (samples, labels) in enumerate(dl):
                print(f'\repoch {e + 1} / {config.classifier_config.epochs}: {(idx + 1) / len(dl): .2%}', end='')
                self.model.zero_grad()
                prediction = self.model(samples)
                loss = cross_entropy(
                    input=prediction,
                    target=labels,
                )
                loss.backward()
                optimizer.step()
        print('')
        self.model.eval()
        self.logger.info('Finished training')

    def predict(self, x: torch.Tensor, use_prob: bool = False) -> torch.Tensor:
        with torch.no_grad():
            prob = self.model(x)
        if use_prob:
            return prob.squeeze(dim=1).detach()
        else:
            return torch.argmax(prob, dim=1)
    
    def get_y_true_y_score(self, dataset: datasets.TeDataset):
        """获取 y_true（真实标签）和 y_score（预测概率）
        
        Args:
            dataset: 测试数据集
            
        Returns:
            y_true: 真实标签的numpy数组
            y_score: 预测概率的numpy数组（每行为一个样本，每列为一个类别的概率）
        """
        self.model = self.model.cpu()
        
        # 从数据集中提取所有样本和标签
        samples_list = []
        labels_list = []
        for i in range(len(dataset)):
            sample, label = dataset[i]
            samples_list.append(sample)
            labels_list.append(label)
        
        # 转换为张量
        all_samples = torch.stack(samples_list).cpu()
        all_labels = torch.stack(labels_list).cpu().numpy()
        
        # 获取预测概率（logits）
        with torch.no_grad():
            prob = self.model(all_samples)
            # 转换为概率（使用softmax）
            y_score = torch.softmax(prob, dim=1).numpy()
        
        self.model = self.model.to(config.device)
        
        return all_labels, y_score
    
    def print_y_true_y_score(self, dataset: datasets.TeDataset, sample_count: int = 10):
        """打印 y_true（真实标签）和 y_score（预测概率）
        
        Args:
            dataset: 测试数据集
            sample_count: 要打印的样本数量，默认为10
        """
        y_true, y_score = self.get_y_true_y_score(dataset)
        
        print("\n" + "="*80)
        print("y_true（真实标签）和 y_score（预测概率）")
        print("="*80)
        
        # 打印样本信息
        print(f"\n总样本数: {len(y_true)}")
        print(f"类别数量: {y_score.shape[1]}")
        print(f"\n打印前 {min(sample_count, len(y_true))} 个样本:")
        print("-"*80)
        
        # 打印表头
        header = f"{'索引':<6} {'y_true':<10}"
        for i in range(y_score.shape[1]):
            header += f" 类别{i}:概率 "
        print(header)
        print("-"*80)
        
        # 打印样本数据
        for i in range(min(sample_count, len(y_true))):
            line = f"{i:<6} {y_true[i]:<10}"
            for j in range(y_score.shape[1]):
                line += f" {y_score[i, j]:.4f}    "
            print(line)
        
        # 打印统计信息
        print("\n统计信息:")
        print(f"y_true 类型: {type(y_true)}, 形状: {y_true.shape}")
        print(f"y_score 类型: {type(y_score)}, 形状: {y_score.shape}")
        print(f"y_true 唯一值: {np.unique(y_true)}")
        print(f"y_score 每行概率和: 均为1.0（softmax归一化）")
        print("="*80 + "\n")

    def test(self, dataset: datasets.TeDataset):
        self.model = self.model.cpu()
        
        # 从数据集中提取所有样本和标签
        samples_list = []
        labels_list = []
        for i in range(len(dataset)):
            sample, label = dataset[i]
            samples_list.append(sample)
            labels_list.append(label)
        
        # 转换为张量
        all_samples = torch.stack(samples_list).cpu()
        all_labels = torch.stack(labels_list).cpu()
        
        predicted_labels = self.predict(all_samples)
        real_labels = all_labels
        
        self.confusion_matrix = metrics.confusion_matrix(
            y_true=real_labels,
            y_pred=predicted_labels,
            labels=[i for i in range(datasets.label_num)]
        )
        self.metrics['Precision'] = metrics.precision_score(
            y_true=real_labels,
            y_pred=predicted_labels,
            average='macro',
            zero_division=0,
        )
        self.metrics['Recall'] = metrics.recall_score(
            y_true=real_labels,
            y_pred=predicted_labels,
            average='macro',
            zero_division=0,
        )
        self.metrics['F1'] = metrics.f1_score(
            y_true=real_labels,
            y_pred=predicted_labels,
            average='macro',
            zero_division=0,
        )
        # 获取每个类别的详细指标
        self.class_metrics = metrics.classification_report(
            y_true=real_labels,
            y_pred=predicted_labels,
            labels=[i for i in range(datasets.label_num)],
            output_dict=True,
            zero_division=0,
        )

        self.model = self.model.to(config.device)

    def binary_test(self, dataset: datasets.TeDataset):
        self.model = self.model.cpu()
        
        # 从数据集中提取所有样本和标签
        samples_list = []
        labels_list = []
        for i in range(len(dataset)):
            sample, label = dataset[i]
            samples_list.append(sample)
            labels_list.append(label)
        
        # 转换为张量
        all_samples = torch.stack(samples_list).cpu()
        all_labels = torch.stack(labels_list).cpu()
        
        predicted_labels = self.predict(all_samples)
        real_labels = all_labels
        for idx, item in enumerate(predicted_labels):
            if item > 0:
                predicted_labels[idx] = 1
        for idx, item in enumerate(real_labels):
            if item > 0:
                real_labels[idx] = 1
        self.confusion_matrix = metrics.confusion_matrix(
            y_true=real_labels,
            y_pred=predicted_labels,
        )
        self.metrics['Precision'] = metrics.precision_score(
            y_true=real_labels,
            y_pred=predicted_labels,
            average='macro',
            zero_division=0,
        )
        self.metrics['Recall'] = metrics.recall_score(
            y_true=real_labels,
            y_pred=predicted_labels,
            average='macro',
            zero_division=0,
        )
        self.metrics['F1'] = metrics.f1_score(
            y_true=real_labels,
            y_pred=predicted_labels,
            average='macro',
            zero_division=0,
        )
        # 获取每个类别的详细指标（二分类）
        self.class_metrics = metrics.classification_report(
            y_true=real_labels,
            y_pred=predicted_labels,
            output_dict=True,
            zero_division=0,
        )

        self.model = self.model.to(config.device)
        
    def print_metrics(self, decimals: int = 4, print_class_metrics: bool = True):
        """打印格式化后的评估指标
        
        Args:
            decimals: 小数位数，默认为4位
            print_class_metrics: 是否打印每个类别的指标，默认为True
        """
        print("整体评估指标:")
        formatted_metrics = {}
        for key, value in self.metrics.items():
            formatted_metrics[key] = round(value, decimals)
        print(formatted_metrics)
        
        if print_class_metrics and self.class_metrics is not None:
            print("\n每个类别的评估指标:")
            # 打印每个类别的指标
            for key, value in self.class_metrics.items():
                if key not in ['accuracy', 'macro avg', 'weighted avg']:
                    try:
                        class_idx = int(key)  # 尝试将key转换为整数（类别索引）
                        print(f"\n类别 {class_idx}:")
                        print(f"  Precision: {round(value['precision'], decimals)}")
                        print(f"  Recall: {round(value['recall'], decimals)}")
                        print(f"  F1-Score: {round(value['f1-score'], decimals)}")
                        print(f"  Support: {value['support']}")
                    except ValueError:
                        # 如果不是整数key（可能是其他统计指标），跳过
                        continue
            
            # 打印宏观平均和加权平均
            print("\n宏观平均:")
            if 'macro avg' in self.class_metrics:
                print(f"  Precision: {round(self.class_metrics['macro avg']['precision'], decimals)}")
                print(f"  Recall: {round(self.class_metrics['macro avg']['recall'], decimals)}")
                print(f"  F1-Score: {round(self.class_metrics['macro avg']['f1-score'], decimals)}")
                print(f"  Support: {self.class_metrics['macro avg']['support']}")
            
            print("\n加权平均:")
            if 'weighted avg' in self.class_metrics:
                print(f"  Precision: {round(self.class_metrics['weighted avg']['precision'], decimals)}")
                print(f"  Recall: {round(self.class_metrics['weighted avg']['recall'], decimals)}")
                print(f"  F1-Score: {round(self.class_metrics['weighted avg']['f1-score'], decimals)}")
                print(f"  Support: {self.class_metrics['weighted avg']['support']}")
            
            if 'accuracy' in self.class_metrics:
                print(f"\nAccuracy: {round(self.class_metrics['accuracy'], decimals)}")
        
    def plot_roc_curve(self, dataset: datasets.TeDataset, is_binary: bool = False):
        """绘制ROC曲线并保存
        
        Args:
            dataset: 测试数据集
            is_binary: 是否为二分类任务
        """
        import matplotlib.pyplot as plt
        import os
        
        self.model = self.model.cpu()
        
        # 从数据集中提取所有样本和标签
        samples_list = []
        labels_list = []
        for i in range(len(dataset)):
            sample, label = dataset[i]
            samples_list.append(sample)
            labels_list.append(label)
        
        # 转换为张量
        all_samples = torch.stack(samples_list).cpu()
        all_labels = torch.stack(labels_list).cpu().numpy()
        
        # 获取预测概率
        with torch.no_grad():
            prob = self.model(all_samples)
            if not is_binary and prob.shape[1] > 2:
                # 多分类任务，使用one-vs-rest策略
                from sklearn.preprocessing import label_binarize
                
                # 将标签二值化
                y_bin = label_binarize(all_labels, classes=[i for i in range(datasets.label_num)])
                n_classes = y_bin.shape[1]
                
                # 计算每个类别的ROC曲线和AUC
                fpr = dict()
                tpr = dict()
                roc_auc = dict()
                
                for i in range(n_classes):
                    fpr[i], tpr[i], _ = metrics.roc_curve(y_bin[:, i], prob[:, i].numpy())
                    roc_auc[i] = metrics.roc_auc_score(y_bin[:, i], prob[:, i].numpy())
                
                # 绘制所有类别的ROC曲线
                plt.figure(figsize=(10, 8))
                colors = ['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple']
                for i, color in zip(range(n_classes), colors[:n_classes]):
                    plt.plot(fpr[i], tpr[i], color=color, lw=2, 
                             label='ROC curve of class {0} (area = {1:0.2f})' 
                             ''.format(i, roc_auc[i]))
            else:
                # 二分类任务
                if prob.shape[1] > 1:
                    # 如果模型输出多个概率，取正类概率（假设类别1为正类）
                    y_score = prob[:, 1].numpy()
                else:
                    # 如果模型输出单个概率，直接使用
                    y_score = prob.numpy()
                    
                # 确保真实标签是二值的（无论是否指定is_binary=True，都需要二值化）
                y_test = np.where(all_labels > 0, 1, 0)
                
                # 计算ROC曲线和AUC
                fpr, tpr, _ = metrics.roc_curve(y_test, y_score)
                roc_auc = metrics.roc_auc_score(y_test, y_score)
                
                # 绘制ROC曲线
                plt.figure(figsize=(10, 8))
                plt.plot(fpr, tpr, color='darkorange', lw=2, 
                         label='ROC curve (area = %0.2f)' % roc_auc)
        
        # 绘制对角线
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        
        # 设置图形属性
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{self.name} Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        # 保存ROC曲线，文件名包含模型名称
        from src import config
        model_name = self.name.replace('_classifier', '')
        roc_path = config.path_config.gan_outs / f'{model_name}_roc_curve_{"binary" if is_binary else "multiclass"}.jpg'
        plt.savefig(roc_path)
        plt.close()
        
        print(f"ROC曲线已保存至: {roc_path}")
        
        self.model = self.model.to(config.device)
    
    def plot_pr_curve(self, dataset: datasets.TeDataset, is_binary: bool = False):
        """绘制PR-AUC曲线并保存
        
        Args:
            dataset: 测试数据集
            is_binary: 是否为二分类任务
        """
        import matplotlib.pyplot as plt
        import os
        
        self.model = self.model.cpu()
        
        # 从数据集中提取所有样本和标签
        samples_list = []
        labels_list = []
        for i in range(len(dataset)):
            sample, label = dataset[i]
            samples_list.append(sample)
            labels_list.append(label)
        
        # 转换为张量
        all_samples = torch.stack(samples_list).cpu()
        all_labels = torch.stack(labels_list).cpu().numpy()
        
        # 获取预测概率
        with torch.no_grad():
            prob = self.model(all_samples)
            if not is_binary and prob.shape[1] > 2:
                # 多分类任务，使用one-vs-rest策略
                from sklearn.preprocessing import label_binarize
                
                # 将标签二值化
                y_bin = label_binarize(all_labels, classes=[i for i in range(datasets.label_num)])
                n_classes = y_bin.shape[1]
                
                # 计算每个类别的PR曲线和AUC
                precision = dict()
                recall = dict()
                pr_auc = dict()
                
                for i in range(n_classes):
                    precision[i], recall[i], _ = metrics.precision_recall_curve(y_bin[:, i], prob[:, i].numpy())
                    pr_auc[i] = metrics.auc(recall[i], precision[i])
                
                # 绘制所有类别的PR曲线
                plt.figure(figsize=(10, 8))
                colors = ['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple']
                for i, color in zip(range(n_classes), colors[:n_classes]):
                    plt.plot(recall[i], precision[i], color=color, lw=2, 
                             label='PR curve of class {0} (area = {1:0.2f})' 
                             ''.format(i, pr_auc[i]))
            else:
                # 二分类任务
                if prob.shape[1] > 1:
                    # 如果模型输出多个概率，取正类概率（假设类别1为正类）
                    y_score = prob[:, 1].numpy()
                else:
                    # 如果模型输出单个概率，直接使用
                    y_score = prob.numpy()
                    
                # 确保真实标签是二值的（无论是否指定is_binary=True，都需要二值化）
                y_test = np.where(all_labels > 0, 1, 0)
                
                # 计算PR曲线和AUC
                precision, recall, _ = metrics.precision_recall_curve(y_test, y_score)
                pr_auc = metrics.auc(recall, precision)
                
                # 绘制PR曲线
                plt.figure(figsize=(10, 8))
                plt.plot(recall, precision, color='darkorange', lw=2, 
                         label='PR curve (area = %0.2f)' % pr_auc)
        
        # 绘制随机猜测线（正类比例）
        if is_binary or prob.shape[1] == 1:
            pos_ratio = np.sum(y_test) / len(y_test)
            plt.plot([0, 1], [pos_ratio, pos_ratio], color='navy', lw=2, linestyle='--',
                     label=f'Random guess (pos ratio = {pos_ratio:.2f})')
        
        # 设置图形属性
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'{self.name} Precision-Recall (PR) Curve')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        
        # 保存PR曲线，文件名包含模型名称
        from src import config
        model_name = self.name.replace('_classifier', '')
        pr_path = config.path_config.gan_outs / f'{model_name}_pr_curve_{"binary" if is_binary else "multiclass"}.jpg'
        plt.savefig(pr_path)
        plt.close()
        
        print(f"PR-AUC曲线已保存至: {pr_path}")
        
        self.model = self.model.to(config.device)
 