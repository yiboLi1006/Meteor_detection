# 该文件实现分类模型的构建、训练、验证、测试,数据来自image_processing_pipeline.py的输出
# 为用于筛选流星图像的Detection传递部分函数,类 


import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, roc_auc_score, confusion_matrix, classification_report, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import glob
import gc
import time
from tqdm import tqdm
import pandas as pd

import torch.backends.cudnn as cudnn
cudnn.benchmark = True


from torchvision import transforms

plt.rcParams["font.family"] = ["Times New Roman"]

# 1. 配置优化
# 修改Config类中的配置
# 在Config类中（约第10-30行）
# 在Config类中添加focal_loss_alpha属性
class Config:
    def __init__(self):
        self.image_size = 750  # 保持图像尺寸不变
        self.batch_size = 32  # 直接设置合适的批次大小
        self.num_workers = 8  # 保持工作进程数
        self.persistent_workers = True
        self.prefetch_factor = 4
        self.pin_memory = True  # 锁定内存加速GPU传输
        self.use_amp = True  # 启用混合精度训练
        self.gradient_accumulation_steps = 2  # 增加梯度累积步数
        self.early_stopping_patience = 10  # 早停策略
        self.lr = 0.0002  # 学习率
        self.weight_decay = 5e-5  # 添加权重衰减以防止过拟合
        self.lr_scheduler_patience = 3  # 学习率调度器耐心值
        self.lr_scheduler_factor = 0.5  # 学习率衰减因子
        self.lr_min = 1e-6  # 添加最小学习率限制
        # Focal Loss参数
        self.focal_loss_gamma = 1.9  # Focal Loss的gamma参数，控制难易样本的权重
        self.focal_loss_alpha = 0.78  # 增加对正类的关注权重
        self.use_focal_loss = True  # 是否使用Focal Loss替代BCEWithLogitsLoss
        self.prediction_threshold = 0.3  # 进一步降低决策阈值以提高正类召回率

        self.use_early_stopping_delta = True  # 使用早停阈值
        self.early_stopping_delta = 0.001  # 早停损失变化阈值

config = Config()

# 数据集类 - 优化大数据集处理
class MeteorDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        # 创建默认的归一化变换
        self.default_transform = transforms.Compose([
            transforms.Normalize(mean=[0.5], std=[0.5])  # 灰度图的归一化参数
        ])
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        # 直接加载预处理后的图像
        try:
            # 确保路径使用UTF-8编码
            image_path = image_path.encode('utf-8').decode('utf-8')
            
            # 检查文件是否存在
            if not os.path.exists(image_path):
                print(f"文件不存在: {image_path}")
                return torch.zeros((1, config.image_size, config.image_size)), label
            
            # 使用PIL加载灰度图并调整大小
            image = Image.open(image_path).convert('L')
            # 对于750x750的图像，不需要再调整大小，但保留代码以兼容不同尺寸
            if image.size != (config.image_size, config.image_size):
                image = image.resize((config.image_size, config.image_size), Image.LANCZOS)
            
            # 转换为numpy数组
            image = np.array(image)
            
            if image is None or image.size == 0:
                print(f"数据为空: {image_path}")
                return torch.zeros((1, config.image_size, config.image_size)), label

            # 扩展维度以符合CNN输入要求 (1, H, W)
            processed_image = np.expand_dims(image, axis=0)

            # 转换为张量并基础归一化（缩放到[0,1]）
            processed_image = torch.from_numpy(processed_image).float() / 255.0

            # 应用高级归一化（标准化）
            processed_image = self.default_transform(processed_image)

            # 如果提供了额外变换，应用它们
            if self.transform:
                processed_image = self.transform(processed_image)

            return processed_image, label
        except Exception as e:
            print(f"读取数据错误: {str(e)}")
            print(f"无法加载数据: {image_path}")
            return torch.zeros((1, config.image_size, config.image_size)), label

# 调整后的CNN模型定义 - 适应750x750图像
class OptimizedMeteorCNN(nn.Module):
    def __init__(self):
        super(OptimizedMeteorCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 64, kernel_size=11, stride=2, padding=5)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)  # 使用inplace以减少内存使用
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        

        self.conv2 = nn.Conv2d(64, 128, kernel_size=7, stride=1, padding=3)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        

        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        

        self.conv4 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(384)
        self.relu4 = nn.ReLU(inplace=True)
        

        self.conv5 = nn.Conv2d(384, 512, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.relu5 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 自适应池化确保输出大小一致，即使输入尺寸有细微差异
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))
        
        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Linear(512 * 8 * 8, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        x = self.pool4(x)
        
        x = self.adaptive_pool(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        
        return x

# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    model.to(device)
    
    # 初始化混合精度训练
    scaler = torch.amp.GradScaler(enabled=config.use_amp)
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    best_val_loss = float('inf')
    early_stopping_counter = 0
    
    # 添加时间戳用于文件命名
    timestamp = time.strftime('%m%d_%H%M')
    
    # 学习率记录
    learning_rates = []
    
    # 创建日志文件 - 添加时间戳
    log_file_path = f'training_log_{timestamp}.txt'
    with open(log_file_path, 'w') as log_file:
        log_file.write('Training Log\n')
        log_file.write('=' * 80 + '\n')
        log_file.write(f'Started training on device: {device}\n')
        log_file.write(f'Timestamp: {timestamp}\n')
        log_file.write(f'Number of epochs: {num_epochs}\n')
        log_file.write(f'Model: OptimizedMeteorCNN\n')
        log_file.write(f'Batch size: {config.batch_size}\n')
        log_file.write(f'Initial learning rate: {config.lr}\n')
        log_file.write(f'Loss function: {"FocalLoss" if config.use_focal_loss else "BCEWithLogitsLoss"}\n')
        log_file.write(f'Early stopping patience: {config.early_stopping_patience}\n')
        log_file.write('=' * 80 + '\n')
        log_file.write('Epoch, Train Loss, Train Accuracy, Val Loss, Val Accuracy, Learning Rate, Best Model Saved, Time\n')
    
    # 添加时间戳到最佳模型保存路径
    best_model_path = f'meteor_detection_best_model_{timestamp}.pth'
    
    for epoch in range(num_epochs):
        # 记录当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        print(f"当前学习率: {current_lr:.8f}")
        
        # 记录每个epoch的开始时间
        epoch_start_time = time.time()
        

        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # 显示训练进度
        train_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - 训练", leave=False)
        

        for i, (images, labels) in enumerate(train_iterator):

            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)

            optimizer.zero_grad()
            
            # 混合精度训练
            with torch.amp.autocast(device_type='cuda', enabled=config.use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            # 梯度累积
            loss = loss / config.gradient_accumulation_steps
                
            scaler.scale(loss).backward()
            
            if (i + 1) % config.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += loss.item() * images.size(0) * config.gradient_accumulation_steps
            predicted = (outputs > 0.5).float()
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)
            
            # 更新进度条
            train_iterator.set_postfix(loss=loss.item() * config.gradient_accumulation_steps, 
                                      accuracy=train_correct/train_total)

        train_loss = train_loss / len(train_loader.dataset)
        train_accuracy = train_correct / train_total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        # 显示验证进度
        val_iterator = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - 验证", leave=False)
        
        with torch.no_grad():
            for images, labels in val_iterator:
                images, labels = images.to(device), labels.to(device).float().unsqueeze(1)

                with torch.amp.autocast(device_type='cuda', enabled=config.use_amp):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                predicted = (outputs > 0.5).float()
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
                
                # 更新进度条
                val_iterator.set_postfix(loss=loss.item(), accuracy=val_correct/val_total)

        val_loss = val_loss / len(val_loader.dataset)
        val_accuracy = val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # 记录epoch运行时间
        epoch_time = time.time() - epoch_start_time
        
        # 检查是否保存了最佳模型
        best_model_saved = False
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), best_model_path)
            print(f"模型已更新并保存到 {best_model_path}")
            best_model_saved = True
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= config.early_stopping_patience:
                print(f"早停机制触发，在第{epoch+1}轮停止训练")
                # 记录早停信息到日志
                with open(log_file_path, 'a') as log_file:
                    log_file.write(f"Early stopping triggered at epoch {epoch+1}\n")
                break
        
        # 将当前epoch信息写入日志文件
        with open(log_file_path, 'a') as log_file:
            log_file.write(f"{epoch+1}, {train_loss:.6f}, {train_accuracy:.6f}, {val_loss:.6f}, {val_accuracy:.6f}, {current_lr:.8f}, {best_model_saved}, {epoch_time:.2f}\n")
        
        print(f'Epoch {epoch+1}/{num_epochs}, '
              f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, '
              f'Time: {epoch_time:.2f}s')
        
        # 学习率调度
        scheduler.step(val_loss)
        
        # 每个epoch后清理内存
        torch.cuda.empty_cache()
        gc.collect()
    
    # 训练结束后，记录最终结果到日志
    with open(log_file_path, 'a') as log_file:
        log_file.write('=' * 80 + '\n')
        log_file.write(f'Training completed after {len(val_losses)} epochs\n')
        log_file.write(f'Best validation loss: {best_val_loss:.6f}\n')
        if len(train_accuracies) > 0 and len(val_accuracies) > 0:
            log_file.write(f'Final train accuracy: {train_accuracies[-1]:.6f}\n')
            log_file.write(f'Final validation accuracy: {val_accuracies[-1]:.6f}\n')
        log_file.write('Training log completed\n')
    
    print(f"训练日志已保存至 {log_file_path}")
    
    # 绘制训练曲线 - 增加学习率曲线
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(range(1, len(train_losses)+1), train_losses, 'b-', label='Training Loss')
    plt.plot(range(1, len(val_losses)+1), val_losses, 'r-', label='Validation Loss')
    plt.title('Loss vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(range(1, len(train_accuracies)+1), train_accuracies, 'b-', label='Training Accuracy')
    plt.plot(range(1, len(val_accuracies)+1), val_accuracies, 'r-', label='Validation Accuracy')
    plt.title('Accuracy vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # 添加学习率曲线
    plt.subplot(2, 2, 3)
    plt.plot(range(1, len(learning_rates)+1), learning_rates, 'g-', label='Learning Rate')
    plt.title('Learning Rate vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.yscale('log')  # 使用对数刻度更好地显示学习率变化
    plt.legend()
    
    # 添加早停标记
    plt.subplot(2, 2, 4)
    plt.plot(range(1, len(val_losses)+1), val_losses, 'r-', label='Validation Loss')
    plt.axvline(x=len(val_losses), color='r', linestyle='--', label='Training Stopped')
    plt.title('Validation Loss with Stopping Point')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'training_curves_enhanced_{timestamp}.png',dpi=300)

    print(f"训练曲线图已保存到 training_curves_{timestamp}.png")
    
    # 加载最佳模型
    model.load_state_dict(torch.load(best_model_path))
    return model, timestamp

# 评估函数 - 针对大数据集优化
# 修改evaluate_model函数中的决策阈值部分(约第337行)
def evaluate_model(model, test_loader, timestamp=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    all_labels = []
    all_predictions = []
    all_probabilities = []

    # 显示进度
    test_iterator = tqdm(test_loader, desc="评估模型", leave=False)
    
    with torch.no_grad():
        for images, labels in test_iterator:
            images = images.to(device)
            outputs = model(images)
            probabilities = outputs.cpu().numpy()
            # 使用配置中的阈值替代固定的0.5
            predicted = (probabilities > config.prediction_threshold).astype(int)
            
            all_labels.extend(labels.numpy())
            all_predictions.extend(predicted)
            all_probabilities.extend(probabilities)

    # 确保数据格式正确
    all_labels = np.array(all_labels).ravel()
    all_predictions = np.array(all_predictions).ravel()
    all_probabilities = np.array(all_probabilities).ravel()

    # 计算传统评估指标
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, zero_division=0)
    recall = recall_score(all_labels, all_predictions, zero_division=0)
    f1 = f1_score(all_labels, all_predictions, zero_division=0)
    
    # Kappa系数
    kappa = cohen_kappa_score(all_labels, all_predictions)
    
    # AUC-ROC分数
    try:
        auc_roc = roc_auc_score(all_labels, all_probabilities)
    except ValueError:
        auc_roc = 0.5  # 在只有一个类别时的默认值
    
    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_predictions)
    
    # 生成分类报告
    class_report = classification_report(all_labels, all_predictions, zero_division=0)

    print(f'评估指标:')
    print(f'准确率: {accuracy:.4f}')
    print(f'精确率: {precision:.4f}')
    print(f'召回率: {recall:.4f}')
    print(f'F1分数: {f1:.4f}')
    print(f'Kappa系数: {kappa:.4f}')
    print(f'AUC-ROC分数: {auc_roc:.4f}')
    print(f'分类报告:\n{class_report}')
    print(f'混淆矩阵:\n{cm}')

    # 绘制混淆矩阵并保存 - 修改为按比例显示
    plt.figure(figsize=(8, 6))
    
    # 计算混淆矩阵的比例
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # 绘制归一化的混淆矩阵（按比例）
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Class')
    plt.ylabel('Actual Class')
    plt.tight_layout()
    
    # 添加时间戳
    if timestamp:
        confusion_matrix_path = f'confusion_matrix_{timestamp}.png'
    else:
        confusion_matrix_path = 'confusion_matrix.png'
    plt.savefig(confusion_matrix_path,dpi=300)

    plt.close()
    print(f"混淆矩阵已保存到 {confusion_matrix_path}")
    
    # 绘制ROC曲线并保存
    try:
        fpr, tpr, _ = roc_curve(all_labels, all_probabilities)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {auc_roc:.3f})')
        plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve') 
        plt.legend(loc="lower right")
        plt.tight_layout()
        
        if timestamp:
            roc_curve_path = f'roc_curve_{timestamp}.png'
        else:
            roc_curve_path = 'roc_curve.png'
        plt.savefig(roc_curve_path,dpi=300)

        plt.close()
        print(f"ROC曲线已保存到 {roc_curve_path}")
    except ValueError:
        print("警告: 无法绘制ROC曲线，可能是因为只有一个类别存在")

    # 保存评估指标到文件
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Kappa Coefficient', 'AUC-ROC Score'], 
        'Value': [accuracy, precision, recall, f1, kappa, auc_roc]
    })
    
    # 添加时间戳
    if timestamp:
        metrics_path = f'evaluation_metrics_{timestamp}.csv'
    else:
        metrics_path = 'evaluation_metrics.csv'
    metrics_df.to_csv(metrics_path, index=False, encoding='utf-8')
    print(f"评估指标已保存到 {metrics_path}")

    return accuracy, precision, recall, f1, kappa, auc_roc


def main():

    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"当前GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA版本: {torch.version.cuda}")

    start_time = time.time()
    
    # 设置数据集路径
    output_dir = r'.\DataSet'

    # 从预处理后的文件夹加载图像路径和标签
    print('加载预处理后的图像...')
    processed_positive_dir = os.path.join(output_dir, 'Positive')
    processed_negative_dir = os.path.join(output_dir, 'Negative')
    print(f"正样本目录: {processed_positive_dir}")
    print(f"负样本目录: {processed_negative_dir}")

    # 检查是否有重复文件
    positive_files = os.listdir(processed_positive_dir)
    negative_files = os.listdir(processed_negative_dir)
    positive_set = set(positive_files)
    negative_set = set(negative_files)
    intersection = positive_set.intersection(negative_set)
    if intersection:
        print(f"警告：发现 {len(intersection)} 个重复文件")


    # Windows中只需要小写扩展名，避免重复计数
    extensions = ['.jpg', '.jpeg', '.png']
    
    image_paths = []
    labels = []
    
    # 加载正样本（标签为1）
    if os.path.exists(processed_positive_dir):
        # 使用glob获取所有匹配的图像文件
        positive_files = []
        for ext in extensions:
            pattern = os.path.join(processed_positive_dir, f'*{ext}')
            positive_files.extend(glob.glob(pattern))
        
        # 添加正样本到列表
        image_paths.extend(positive_files)
        labels.extend([1] * len(positive_files))
        
        print(f'加载了 {len(positive_files)} 个正样本')
    else:
        print(f'警告：正样本文件夹不存在: {processed_positive_dir}')
    
    # 加载负样本（标签为0）
    if os.path.exists(processed_negative_dir):
        negative_files = []
        for ext in extensions:
            pattern = os.path.join(processed_negative_dir, f'*{ext}')
            negative_files.extend(glob.glob(pattern))


        image_paths.extend(negative_files)
        labels.extend([0] * len(negative_files))
        
        print(f'加载了 {len(negative_files)} 个负样本')
    else:
        print(f'警告：负样本文件夹不存在: {processed_negative_dir}')
    
    # 检查是否加载到了数据
    if len(image_paths) == 0:
        print("错误：未加载到任何图像数据！请检查数据集路径是否正确。")
        return
    
    print(f'总共加载了 {len(image_paths)} 个图像')
    
    # 检查是否有足够的数据用于训练
    if len(image_paths) < 100:
        print("警告：数据集较小，可能影响模型训练效果。")
    
    # 记录数据加载时间
    data_loading_time = time.time() - start_time
    print(f'数据加载耗时: {data_loading_time:.2f} 秒')
    
    print('划分数据集...')
    # 划分训练集、验证集和测试集
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
    )
    
    print(f'数据集划分完成：训练集 {len(X_train)}，验证集 {len(X_val)}，测试集 {len(X_test)}')
    
    # 清理不再需要的变量以节省内存
    del image_paths, labels, X_train_val, y_train_val
    gc.collect()

    # 创建数据集
    train_dataset = MeteorDataset(X_train, y_train)
    val_dataset = MeteorDataset(X_val, y_val)
    test_dataset = MeteorDataset(X_test, y_test)

    # 创建数据加载器
    print('创建数据加载器...')
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor,
        pin_memory=config.pin_memory,
        persistent_workers=True  # 添加这一行
    )
    

    val_loader = DataLoader(val_dataset, 
                           batch_size=config.batch_size * 2,
                           shuffle=False, 
                           num_workers=config.num_workers, 
                           pin_memory=config.pin_memory)
    
    test_loader = DataLoader(test_dataset, 
                            batch_size=config.batch_size * 2, 
                            shuffle=False, 
                            num_workers=config.num_workers, 
                            pin_memory=config.pin_memory)

   
    print('创建模型...')
    # 初始化模型、损失函数和优化器
    model = OptimizedMeteorCNN()
    
    # 计算正样本权重以解决类别不平衡问题
    positive_count = len(positive_files)
    negative_count = len(negative_files)
    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

    if config.use_focal_loss:
        alpha_tensor = torch.tensor([config.focal_loss_alpha], device=device)
        criterion = FocalLoss(alpha=alpha_tensor, gamma=config.focal_loss_gamma)
        print(f"使用Focal Loss，alpha={config.focal_loss_alpha:.4f}, gamma={config.focal_loss_gamma}")
    else:
        pos_weight = torch.tensor([negative_count / positive_count], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"使用BCEWithLogitsLoss，pos_weight={pos_weight.item():.4f}")

    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    
    # 学习率调度器 - 支持最小学习率
    if hasattr(config, 'lr_scheduler_patience'):
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                       patience=config.lr_scheduler_patience, 
                                                       factor=config.lr_scheduler_factor, 
                                                       min_lr=config.lr_min if hasattr(config, 'lr_min') else 0)
    else:
        # 如果配置中没有相关参数，使用默认值
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                       patience=3, 
                                                       factor=0.1)
    
    # 训练模型
    training_start_time = time.time()
    trained_model, timestamp = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs)
    training_time = time.time() - training_start_time
    print(f'模型训练耗时: {training_time/60:.2f} 分钟')

    # 保存最终模型
    final_model_path = f'meteor_detection_final_model_{timestamp}.pth'
    torch.save(trained_model.state_dict(), final_model_path)
    print(f'最终模型已保存至 {final_model_path}')

    # 评估模型
    print('评估模型...')
    evaluate_model(trained_model, test_loader, timestamp)
    
    # 计算总耗时
    total_time = time.time() - start_time
    print(f'总执行时间: {total_time/60:.2f} 分钟')

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        # 计算二分类交叉熵损失
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)
        
        # 计算预测概率
        pt = torch.exp(-bce_loss)
        
        # 应用Focal Loss的权重
        focal_loss = ((1 - pt) ** self.gamma) * bce_loss
        
        # 应用类别权重（如果提供）
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_loss = alpha_t * focal_loss
        
        # 根据reduction参数进行聚合
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

if __name__ == '__main__':
    num_epochs=200
    main()