import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import threading
import queue
import time
from sklearn.neighbors import NearestNeighbors

# 配置参数
class Config:
    def __init__(self):
        self.image_size = 750  # 图像尺寸为750x750
        self.num_threads = 8   # 线程数量，可根据CPU核心数调整
        self.target_size = "double"  # 设置为'double'表示将正样本数量翻倍
        self.smote_k_neighbors = 5  # SMOTE算法中使用的近邻数量
        self.output_dir = "DataSet/Positive_oversampled"  # 生成的过采样图像保存目录
        self.batch_size = 500  # 每批生成的样本数量，根据内存情况调整
config = Config()
config.num_samples_to_generate = 1000

# 创建输出目录
def create_output_directory():
    """创建输出目录，如果不存在则创建"""
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)
        print(f"创建输出目录: {config.output_dir}")

# 加载图像数据
def load_images_from_directory(directory):
    """从指定目录加载所有图像并转换为numpy数组"""
    image_files = [f for f in os.listdir(directory) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    images = []
    filenames = []
    
    print(f"正在加载 {directory} 中的图像...")
    for filename in tqdm(image_files, desc="加载图像"):
        try:
            image_path = os.path.join(directory, filename)
            # 打开图像并转换为灰度
            img = Image.open(image_path).convert('L')
            # 确保图像尺寸为750x750
            if img.size != (config.image_size, config.image_size):
                img = img.resize((config.image_size, config.image_size), Image.LANCZOS)
            # 转换为numpy数组并归一化
            img_array = np.array(img) / 255.0
            images.append(img_array)
            filenames.append(filename)
        except Exception as e:
            print(f"加载图像 {filename} 时出错: {str(e)}")
    
    return np.array(images), filenames

# 自定义SMOTE实现，专为图像处理优化
def custom_image_smote(images, num_samples, k_neighbors=5):
    """
    为图像数据实现SMOTE过采样
    images: 原始图像数组，形状为 (n_samples, height, width)
    num_samples: 要生成的新样本数量
    k_neighbors: 用于生成合成样本的近邻数量
    """
    n_samples, height, width = images.shape
    
    # 重塑图像为二维数组以用于SMOTE
    flattened_images = images.reshape(n_samples, -1)
    
    # 初始化最近邻模型
    nn = NearestNeighbors(n_neighbors=k_neighbors, metric='euclidean')
    nn.fit(flattened_images)
    
    # 存储生成的合成样本
    synthetic_samples = []
    
    # 使用tqdm显示进度
    with tqdm(total=num_samples, desc="生成合成样本") as pbar:
        # 为每个需要生成的样本选择一个随机的原始样本
        for i in range(num_samples):
            # 随机选择一个样本作为基础
            base_idx = np.random.randint(0, n_samples)
            base_image = flattened_images[base_idx]
            
            # 找到k个最近邻
            distances, indices = nn.kneighbors([base_image])
            
            # 从k个最近邻中随机选择一个
            neighbor_idx = indices[0][np.random.randint(1, k_neighbors)]  # 排除自己
            neighbor_image = flattened_images[neighbor_idx]
            
            # 在基础样本和选定的近邻之间进行随机插值
            alpha = np.random.random()
            synthetic_flat = base_image + alpha * (neighbor_image - base_image)
            
            # 重塑回原始图像形状
            synthetic_image = synthetic_flat.reshape(height, width)
            
            # 添加到结果列表
            synthetic_samples.append(synthetic_image)
            pbar.update(1)
    
    return np.array(synthetic_samples)

# 保存图像的工作线程函数
def save_image_worker(image_queue, lock):
    """工作线程函数，用于保存图像"""
    while True:
        # 从队列中获取任务
        item = image_queue.get()
        if item is None:  # 终止信号
            break
        
        image_array, filename = item
        
        try:
            # 将图像数据从[0, 1]范围转换回[0, 255]范围
            img_uint8 = (image_array * 255).astype(np.uint8)
            # 创建PIL图像
            img = Image.fromarray(img_uint8)
            
            # 使用线程锁确保文件名唯一性
            with lock:
                output_path = os.path.join(config.output_dir, filename)
                # 保存图像
                img.save(output_path, 'JPEG')
        except Exception as e:
            print(f"保存图像 {filename} 时出错: {str(e)}")
        finally:
            # 标记任务完成
            image_queue.task_done()

# 保存生成的合成图像
def save_synthetic_images(synthetic_images, start_index=0):
    """保存生成的合成图像到输出目录"""
    # 创建图像队列和线程锁
    image_queue = queue.Queue()
    lock = threading.Lock()
    
    # 创建并启动工作线程
    threads = []
    for _ in range(config.num_threads):
        thread = threading.Thread(target=save_image_worker, args=(image_queue, lock))
        thread.daemon = True
        thread.start()
        threads.append(thread)
    
    print(f"开始保存 {len(synthetic_images)} 张合成图像...")
    
    # 向队列添加任务
    for i, synthetic_image in enumerate(tqdm(synthetic_images, desc="添加保存任务")):
        filename = f"synthetic_{start_index + i}.jpg"
        image_queue.put((synthetic_image, filename))
    
    # 等待所有任务完成
    image_queue.join()
    
    # 发送终止信号给所有线程
    for _ in range(config.num_threads):
        image_queue.put(None)
    
    # 等待所有线程结束
    for thread in threads:
        thread.join()
    
    print(f"所有合成图像已保存到 {config.output_dir}")

# 主函数

def main():
    start_time = time.time()
    
    # 创建输出目录
    create_output_directory()
    
    # 加载正样本图像
    positive_dir = "DataSet/original_Positive"
    positive_images, positive_filenames = load_images_from_directory(positive_dir)
    original_positive_count = len(positive_images)
    print(f"加载了 {original_positive_count} 张正样本图像")
    
    # 确定需要生成的正样本数量
    if config.num_samples_to_generate is not None and config.num_samples_to_generate > 0:
        # 如果直接指定了要生成的样本数量，优先使用这个值
        num_samples_to_generate = config.num_samples_to_generate
        print(f"目标: 生成 {num_samples_to_generate} 个新样本")
    elif config.target_size == "double":
        # 将正样本数量翻倍
        num_samples_to_generate = original_positive_count
        print(f"目标: 将正样本数量翻倍")
    elif config.target_size is None:
        # 默认情况下，将正样本数量增加到与负样本相同
        # 仅统计负样本数量，不加载图像到内存
        negative_dir = "DataSet/Negative"
        negative_files = [f for f in os.listdir(negative_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        negative_count = len(negative_files)
        print(f"检测到 {negative_count} 张负样本图像")
        num_samples_to_generate = max(0, negative_count - original_positive_count)
    else:
        num_samples_to_generate = max(0, config.target_size - original_positive_count)
    
    if num_samples_to_generate <= 0:
        print("不需要生成新的正样本，因为正样本数量已经足够或超过目标数量")
        return
    
    print(f"需要生成 {num_samples_to_generate} 张新的正样本图像")
    print(f"分批生成: 每批 {config.batch_size} 个样本")
    
    # 分批次生成和保存样本
    total_generated = 0
    batch_number = 1
    remaining_samples = num_samples_to_generate
    
    while remaining_samples > 0:
        # 计算当前批次的样本数量
        current_batch_size = min(config.batch_size, remaining_samples)
        
        print(f"\n--- 批次 {batch_number}: 生成 {current_batch_size} 个样本 ---\n")
        
        # 生成当前批次的样本
        batch_synthetic_images = custom_image_smote(
            positive_images, 
            current_batch_size, 
            k_neighbors=config.smote_k_neighbors
        )
        
        # 保存当前批次的样本
        if len(batch_synthetic_images) > 0:
            save_synthetic_images(batch_synthetic_images, start_index=total_generated)
            total_generated += len(batch_synthetic_images)
            remaining_samples -= len(batch_synthetic_images)
            
            # 显示批次完成信息
            print(f"批次 {batch_number} 完成: 已生成 {total_generated}/{num_samples_to_generate} 个样本")
        
        batch_number += 1
    
    # 计算总耗时
    total_time = time.time() - start_time
    print(f"\nSMOTE过采样完成！")
    print(f"原始正样本数量: {original_positive_count}")
    print(f"生成的正样本数量: {total_generated}")
    print(f"过采样后总正样本数量: {original_positive_count + total_generated}")
    print(f"总耗时: {total_time:.2f} 秒")

if __name__ == "__main__":
    main()