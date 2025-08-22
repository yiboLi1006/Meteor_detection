import os
import sys
import shutil
import torch
import concurrent.futures
from tqdm import tqdm
import multiprocessing
import time
from torchvision import transforms


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Train_val__test_meteor_detection_cnn import Config, OptimizedMeteorCNN, MeteorDataset, Dataset
from image_processing_pipeline import process_image, split

# 创建一个自定义数据集类，直接使用PIL图像对象
class InMemoryImageDataset(Dataset):
    def __init__(self, images, transform=None):
        self.images = images
        # 如果没有提供变换，创建一个默认的变换（将PIL图像转换为张量）
        self.transform = transform if transform is not None else transforms.ToTensor()
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = self.images[idx]
        
        # 将PIL图像转换为张量
        if self.transform:
            img = self.transform(img)
        
        return img

# 加载模型
def load_model(model_path, device):
    try:
        config = Config()
        model = OptimizedMeteorCNN()
        
        # 加载模型权重
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        config.prediction_threshold = 0.2
        print(f"成功加载模型: {model_path}")
        return model, config
    except Exception as e:
        print(f"加载模型失败: {e}")
        return None, None

# 处理单个图像并检测流星
def process_and_detect(image_file, input_folder, output_folder, model, config, device):
    try:
        input_path = os.path.join(input_folder, image_file)
        base_name = os.path.splitext(image_file)[0]
        
        config.prediction_threshold = 0.2
        
        # 1. 预处理图像
        processed_img = process_image(input_path)
        if processed_img is None:
            print(f"   图像{image_file}预处理失败，跳过")
            return False, base_name, 0.0
        
        # 2. 在内存中分割图像为4个子图
        split_images_with_labels = split(processed_img)
        if not split_images_with_labels:
            print(f"   图像{image_file}分割失败，跳过")
            return False, base_name, 0.0
        
        sub_images = []
        for i, (img, label) in enumerate(split_images_with_labels):
            sub_images.append(img)
        
        # 3. 创建图像预处理变换（添加归一化）
        preprocess_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        # 4. 创建内存数据集和数据加载器
        dataset = InMemoryImageDataset(sub_images, transform=preprocess_transform)
        
        cpu_count = multiprocessing.cpu_count()
        optimal_workers = min(max(1, cpu_count // 2), 4)
        
        data_loader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=config.batch_size, 
            shuffle=False, 
            num_workers=optimal_workers,  
            pin_memory=config.pin_memory
        )

        # 4. 使用模型进行预测
        has_meteor = False
        max_confidence = 0.0  # 记录最大置信度
        with torch.no_grad():
            for images in data_loader:
                images = images.to(device)
                outputs = model(images)
                probabilities = torch.sigmoid(outputs).cpu().numpy().flatten()
                
                # 记录所有子图中的最大置信度
                if len(probabilities) > 0:
                    current_max = max(probabilities)
                    if current_max > max_confidence:
                        max_confidence = current_max
                
                # 检查每个子图是否预测为流星
                for prob in probabilities:
                    if prob >= config.prediction_threshold:
                        has_meteor = True
                        break  # 一旦发现流星，即可标记整图
                if has_meteor:
                    break  # 如果已发现流星，提前结束循环
        
        return has_meteor, base_name, max_confidence  # 返回最大置信度
    except Exception as e:
        print(f"处理图像{image_file}时发生错误: {e}")
        return False, base_name, 0.0

# 主检测函数
def detect_meteors(input_folder, output_folder, model_path, max_workers=None):
    # 检查输入文件夹是否存在
    if not os.path.exists(input_folder):
        print(f"错误：输入文件夹不存在: {input_folder}")
        return False
    
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)
    
    # 确定设备（GPU或CPU）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    model, config = load_model(model_path, device)
    if model is None:
        return False
    
    # 获取输入文件夹中的所有图像文件
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        print(f"错误：在{input_folder}中未找到图像文件")
        return False
    
    total_images = len(image_files)
    print(f"找到{total_images}张图片待处理")
    
    # 设置最大工作线程数
    if max_workers is None:
        cpu_count = multiprocessing.cpu_count()
        max_workers = min(10, cpu_count)  # 设置上限为10
    
    print(f"使用多线程处理，最大工作线程数: {max_workers}")
    
    # 统计信息
    processed_count = 0
    detected_count = 0
    detected_files = []
    file_confidence_map = {}  # 存储文件名和对应的置信度
    
    # 使用线程池并行处理图像
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_image = {
            executor.submit(process_and_detect, image_file, input_folder, output_folder, model, config, device): image_file 
            for image_file in image_files
        }
        
        # 使用tqdm显示进度条
        for future in tqdm(concurrent.futures.as_completed(future_to_image), total=total_images, desc="检测进度"):
            image_file = future_to_image[future]
            try:
                has_meteor, base_name, confidence = future.result()  # 获取置信度
                processed_count += 1
                
                if has_meteor:
                    detected_count += 1
                    detected_files.append(image_file)
                    file_confidence_map[image_file] = confidence  # 保存置信度
                    
                    # 复制原始图像到结果文件夹
                    src_path = os.path.join(input_folder, image_file)
                    dst_path = os.path.join(output_folder, image_file)
                    shutil.copy2(src_path, dst_path)
                    print(f"   检测到流星轨迹: {image_file} (置信度: {confidence:.4f})")
            except Exception as e:
                print(f"处理图像{image_file}时发生意外错误: {e}")
    
    print(f"\n流星检测完成！")
    print(f"成功处理了{processed_count}张图片")
    print(f"检测到{detected_count}张包含流星轨迹的图片")
    print(f"检测结果已保存到: {output_folder}")
    
    # 保存检测结果列表，包含置信度信息
    if detected_files:
        result_list_path = os.path.join(output_folder, 'detected_files.txt')
        with open(result_list_path, 'w', encoding='utf-8') as f:
            # 写入模型基础参数信息
            f.write("====== 模型基础参数信息 ======\n")
            f.write(f"预测阈值: {config.prediction_threshold}\n")
            f.write(f"批次大小: {config.batch_size}\n")
            f.write(f"使用设备: {device}\n")
            if hasattr(config, 'focal_loss_gamma'):
                f.write(f"Focal Loss Gamma: {config.focal_loss_gamma}\n")
            if hasattr(config, 'focal_loss_alpha'):
                f.write(f"Focal Loss Alpha: {config.focal_loss_alpha}\n")
            if hasattr(config, 'image_size'):
                f.write(f"图像尺寸: {config.image_size}\n")
            f.write(f"模型路径: {model_path}\n")
            f.write(f"检测时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=============================\n\n")
            
            # 写入表头
            f.write("文件名,置信度\n")
            # 写入每个检测到的文件及其置信度
            for file in detected_files:
                confidence = file_confidence_map.get(file, 0.0)
                f.write(f"{file},{confidence:.4f}\n")
        print(f"检测到的文件列表（含置信度和模型参数）已保存到: {result_list_path}")
    
    return True

# 主函数
if __name__ == "__main__":
    # 定义各文件夹路径（使用相对路径）
    INPUT_FOLDER = "./Camera_jpg_data"
    OUTPUT_FOLDER = "./result"
    MODEL_PATH = "./model.pth"
    
    # 记录开始时间
    start_time = time.time()
    
    print("=====================================================")
    print("                  流星检测系统启动                   ")
    print("=====================================================")
    print(f"输入文件夹: {INPUT_FOLDER}")
    print(f"输出文件夹: {OUTPUT_FOLDER}")
    print(f"模型路径: {MODEL_PATH}")
    
    # 执行流星检测
    success = detect_meteors(
        INPUT_FOLDER,
        OUTPUT_FOLDER,
        MODEL_PATH
    )
    
    # 记录结束时间
    end_time = time.time()
    total_time = end_time - start_time
    
    if success:
        print(f"\n流星检测成功完成！总耗时: {total_time:.2f} 秒")
    else:

        print(f"\n流星检测执行失败，请查看错误信息。总耗时: {total_time:.2f} 秒")
