import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time
import glob
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, roc_auc_score, confusion_matrix, classification_report, roc_curve
import gc

# 引用现有的meteor_detection_cnn.py中的类和函数
from Train_val__test_meteor_detection_cnn import Config, OptimizedMeteorCNN, MeteorDataset

# 直接导入evaluate_model函数，确保正确引用
try:
    from Train_val__test_meteor_detection_cnn import evaluate_model
    print("成功导入evaluate_model函数")
except ImportError as e:
    print(f"导入evaluate_model函数失败: {e}")
    # 如果导入失败，定义一个简化版的evaluate_model函数作为备选
    def evaluate_model(model, data_loader, timestamp=None):
        print("使用备选的evaluate_model函数")
        device = next(model.parameters()).device
        model.eval()
        all_labels = []
        all_predictions = []
        all_probabilities = []
        
        with torch.no_grad():
            for images, labels in data_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                probabilities = torch.sigmoid(outputs).detach().cpu().numpy()
                predictions = (probabilities >= Config().prediction_threshold).astype(int)
                
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predictions.flatten())
                all_probabilities.extend(probabilities.flatten())
        
        # 计算评估指标
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, zero_division=0)
        recall = recall_score(all_labels, all_predictions, zero_division=0)
        f1 = f1_score(all_labels, all_predictions, zero_division=0)
        kappa = cohen_kappa_score(all_labels, all_predictions)
        
        # 计算AUC-ROC
        try:
            auc_roc = roc_auc_score(all_labels, all_probabilities)
        except ValueError:
            auc_roc = 0.5
        
        # 绘制并保存混淆矩阵
        cm = confusion_matrix(all_labels, all_predictions)
        print(f'混淆矩阵:\n{cm}')
        
        # 绘制归一化的混淆矩阵
        plt.figure(figsize=(8, 6))
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', cbar=False)
        plt.title('Confusion Matrix (Normalized)')
        plt.xlabel('Predicted Class')
        plt.ylabel('Actual Class')
        plt.tight_layout()
        
        # 添加时间戳到保存的文件名
        if timestamp:
            confusion_matrix_path = f'confusion_matrix_{timestamp}.png'
        else:
            confusion_matrix_path = 'confusion_matrix.png'
        plt.savefig(confusion_matrix_path)
        plt.close()
        print(f"混淆矩阵已保存到 {confusion_matrix_path}")
        
        return accuracy, precision, recall, f1, kappa, auc_roc

# 设置中文字体支持
plt.rcParams["font.family"] = ["Times New Roman"]

# 确保中文显示正常
plt.rcParams['axes.unicode_minus'] = False

# 配置验证参数
class ValidateConfig:
    def __init__(self):
        self.config = Config()
        # 可以根据需要覆盖原始配置
        self.batch_size = self.config.batch_size * 2  # 验证时使用更大的批次
        self.num_workers = self.config.num_workers
        self.pin_memory = self.config.pin_memory
        self.prediction_threshold = self.config.prediction_threshold

validate_config = ValidateConfig()

# 加载模型函数
def load_model(model_path):
    """
    加载预训练模型
    Args:
        model_path: 模型文件路径
    Returns:
        加载好权重的模型
    """
    print(f"正在加载模型: {model_path}")
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    # 创建模型实例
    model = OptimizedMeteorCNN()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # 设置为评估模式
    
    print("模型加载成功，已切换到评估模式")
    return model, device

# 准备验证数据函数 - 修改为只使用测试集
def prepare_validation_data(data_dir=None):
    """
    准备验证数据集 - 只使用与原始训练相同划分的测试集
    Args:
        data_dir: 数据集目录路径
    Returns:
        数据加载器（仅测试集）
    """
    if data_dir is None:
        # 使用默认的数据集路径
        data_dir = os.path.join('.', 'DataSet')
        
    print(f"使用数据集目录: {data_dir}")
    
    # 检查数据集目录是否存在
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"数据集目录不存在: {data_dir}")
    
    # 支持的图像扩展名
    extensions = ['.jpg', '.jpeg', '.png']
    
    # 加载所有图像路径和标签
    image_paths = []
    labels = []
    
    # 遍历正负样本文件夹
    for class_dir, label in [('Positive', 1), ('Negative', 0)]:
        class_path = os.path.join(data_dir, class_dir)
        if not os.path.exists(class_path):
            print(f"警告: 类别目录不存在: {class_path}")
            continue
        
        # 收集所有图像文件
        class_files = []
        for ext in extensions:
            pattern = os.path.join(class_path, f'*{ext}')
            class_files.extend(glob.glob(pattern))
        
        # 添加到列表
        image_paths.extend(class_files)
        labels.extend([label] * len(class_files))
    
    if len(image_paths) == 0:
        raise ValueError("未找到任何图像文件，请检查数据集路径")
    
    print(f"共加载了 {len(image_paths)} 张图像")
    print(f"正样本数量: {labels.count(1)}, 负样本数量: {labels.count(0)}")
    
    # 按照原始项目中的方式划分数据集
    print('按原始训练比例划分数据集...')
    # 先划分训练+验证集和测试集 (80% + 20%)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f'数据集划分完成：只使用测试集 {len(X_test)} 张图像进行验证')
    
    # 清理不再需要的变量以节省内存
    del image_paths, labels, X_train_val, y_train_val
    gc.collect()
    
    # 创建测试集数据集
    test_dataset = MeteorDataset(X_test, y_test)
    
    # 创建数据加载器
    test_loader = DataLoader(
        test_dataset,
        batch_size=validate_config.batch_size,
        shuffle=False,
        num_workers=validate_config.num_workers,
        pin_memory=validate_config.pin_memory
    )
    
    return test_loader

# 自定义验证函数，确保包含混淆矩阵输出
def custom_validate(model, data_loader, device, timestamp=None):
    """
    自定义验证函数，提供更灵活的验证选项，并确保混淆矩阵输出
    Args:
        model: 加载好的模型
        data_loader: 数据加载器
        device: 运行设备
        timestamp: 时间戳，用于保存文件命名
    Returns:
        评估指标字典
    """
    model.eval()
    all_labels = []
    all_predictions = []
    all_probabilities = []
    
    start_time = time.time()
    
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # 前向传播
            outputs = model(images)
            
            # 应用sigmoid得到概率
            probabilities = torch.sigmoid(outputs).detach().cpu().numpy()
            
            # 根据阈值进行预测
            predictions = (probabilities >= validate_config.prediction_threshold).astype(int)
            
            # 收集结果
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.flatten())
            all_probabilities.extend(probabilities.flatten())
    
    validation_time = time.time() - start_time
    print(f"验证耗时: {validation_time:.2f} 秒")
    
    # 计算评估指标
    metrics = {
        'accuracy': accuracy_score(all_labels, all_predictions),
        'precision': precision_score(all_labels, all_predictions, zero_division=0),
        'recall': recall_score(all_labels, all_predictions, zero_division=0),
        'f1_score': f1_score(all_labels, all_predictions, zero_division=0),
        'kappa': cohen_kappa_score(all_labels, all_predictions),
    }
    
    # 计算AUC-ROC分数（如果可能的话）
    try:
        metrics['auc_roc'] = roc_auc_score(all_labels, all_probabilities)
    except ValueError:
        metrics['auc_roc'] = 0.5  # 在只有一个类别时的默认值
    
    # 计算并显示混淆矩阵
    print("\n生成混淆矩阵...")
    cm = confusion_matrix(all_labels, all_predictions)
    print(f'混淆矩阵:\n{cm}')
    
    # 绘制混淆矩阵并保存 - 按比例显示
    plt.figure(figsize=(8, 6))
    
    # 计算混淆矩阵的比例
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # 绘制归一化的混淆矩阵（按比例）
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix (Normalized)')
    plt.xlabel('Predicted Class')
    plt.ylabel('Actual Class')
    plt.tight_layout()
    
    # 添加时间戳到保存的文件名
    if timestamp:
        confusion_matrix_path = f'confusion_matrix_custom_{timestamp}.png'
    else:
        confusion_matrix_path = 'confusion_matrix_custom.png'
    plt.savefig(confusion_matrix_path)
    plt.close()
    print(f"混淆矩阵已保存到 {confusion_matrix_path}")
    
    # 打印评估结果
    print("\n评估结果:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")
    
    return metrics, all_labels, all_predictions, all_probabilities

# 主验证函数
def validate_model_performance(model_path, data_dir=None):
    """
    验证模型性能的主函数
    Args:
        model_path: 模型文件路径
        data_dir: 数据集目录路径（可选）
    """
    try:
        # 记录开始时间
        start_time = time.time()
        
        # 加载模型
        model, device = load_model(model_path)
        
        # 准备测试数据集 - 现在只返回测试集数据加载器
        test_loader = prepare_validation_data(data_dir)
        
        # 生成时间戳用于保存结果
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        print(f"当前时间戳: {timestamp}")
        
        # 使用原项目中的evaluate_model函数进行验证
        print("\n===== 使用原始evaluate_model函数进行评估 ======")
        try:
            accuracy, precision, recall, f1, kappa, auc_roc = evaluate_model(model, test_loader, timestamp)
        except Exception as e:
            print(f"调用原始evaluate_model函数出错: {e}")
            print("将使用自定义验证函数")
            # 使用自定义验证函数
            metrics, _, _, _ = custom_validate(model, test_loader, device, timestamp)
        
        # 额外使用自定义验证函数确保混淆矩阵输出
        print("\n===== 使用自定义验证函数进行额外评估 ======")
        metrics, _, _, _ = custom_validate(model, test_loader, device, timestamp)
        
        # 计算总耗时
        total_time = time.time() - start_time
        print(f"\n总验证耗时: {total_time:.2f} 秒")
        print(f"\n验证完成，请查看生成的混淆矩阵图像文件")
        
    except Exception as e:
        print(f"验证过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':

    model_path = r'./meteor_detection_best_model_0821_1923.pth'
    validate_model_performance(model_path)