# 该脚本用于处理图像,最终产生对单张图像四分割的图像,用于模型训练
# 脚本中定义的函数也将被引用,实现对未来待分类图像的预处理 

import os
import sys
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
import concurrent.futures
from tqdm import tqdm

# 确保可以导入同一目录下的模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 定义图像处理函数 - 整合自star_segmentation.py
def process_image(image_path):
    try:
        # 加载图片
        img = Image.open(image_path)
        
        # 步骤1: 转换为灰度图
        img_gray = img.convert('L')
        
        # 步骤2: 降噪处理 - 使用高斯模糊去除噪点
        img_denoised = img_gray.filter(ImageFilter.GaussianBlur(radius=0.7))
        
        # 转换为numpy数组以便进行更精细的图像处理
        img_array = np.array(img_denoised)
        
        # 步骤3: 减小高光部分
        highlight_threshold = 190
        highlight_factor = 0.75
        img_array = np.where(img_array > highlight_threshold, 
                            img_array * highlight_factor, img_array).astype(np.uint8)
        
        # 步骤4: 减小阴影部分
        shadow_threshold = 90
        shadow_factor = 1.4
        img_array = np.where(img_array < shadow_threshold, 
                            img_array * shadow_factor, img_array).astype(np.uint8)
        
        # 步骤5: 减小白色色阶
        white_level = 200
        img_array = np.clip(img_array * (white_level / 255), 0, white_level).astype(np.uint8)
        
        # 步骤6: 转换回PIL图像，进行最后的对比度调整
        img_processed = Image.fromarray(img_array)
        enhancer = ImageEnhance.Contrast(img_processed)
        img_enhanced = enhancer.enhance(1.5)
        
        # 步骤7: 使用阈值分割
        final_array = np.array(img_enhanced)
        threshold = 90
        segmented_array = np.where(final_array > threshold, 255, 0).astype(np.uint8)
        
        # 转换回PIL图像
        segmented_img = Image.fromarray(segmented_array)
        
        # 使用Image.LANCZOS作为重采样方法，这是一种高质量的重采样算法
        target_size = (1500, 1500)
        segmented_img_resized = segmented_img.resize(target_size, Image.LANCZOS)
        
        return segmented_img_resized
    except Exception as e:
        print(f"处理图片{image_path}时出错: {e}")
        return None

# 定义在内存中分割图像的函数
def split(img):
    # 内部辅助函数，提供基础的图像分割逻辑
    def _split_image_base(img_internal):
        try:
            width, height = img_internal.size
            
            # 计算每张子图的尺寸
            half_width = width // 2
            half_height = height // 2
            
            # 定义分割区域
            regions = [
                (0, 0, half_width, half_height),          # 左上
                (half_width, 0, width, half_height),       # 右上
                (0, half_height, half_width, height),      # 左下
                (half_width, half_height, width, height)   # 右下
            ]
            
            # 定义区域标识符
            region_labels = ['top_left', 'top_right', 'bottom_left', 'bottom_right']
            
            return width, height, regions, region_labels
        except Exception as e:
            print(f"图像分割基础处理时出错: {e}")
            return None, None, [], []
    
    try:
        # 调用内部辅助函数获取分割信息
        _, _, regions, region_labels = _split_image_base(img)
        
        if not regions:
            return []
            
        # 分割图片并返回PIL Image对象列表
        split_images = []
        for i, region in enumerate(regions):
            # 裁剪图片
            cropped_img = img.crop(region)
            # 存储裁剪后的PIL Image对象和对应的标签
            split_images.append((cropped_img, region_labels[i]))
            
        return split_images
    except Exception as e:
        print(f"在内存中分割图片时出错: {e}")
        return []

def transformations(img, base_name):
    try:
        # 定义要应用的变换操作
        transformations = {
            'flip_vertical': lambda img: img.transpose(Image.FLIP_TOP_BOTTOM),  # 上下翻转
            'flip_horizontal': lambda img: img.transpose(Image.FLIP_LEFT_RIGHT),  # 左右翻转
            'rotate_90': lambda img: img.transpose(Image.ROTATE_90),  # 顺时针旋转90度
            'rotate_270': lambda img: img.transpose(Image.ROTATE_270),  # 逆时针旋转90度
            'rotate_180': lambda img: img.transpose(Image.ROTATE_180),  # 顺时针旋转180度
        }
        
        # 应用每种变换并返回变换后的图像和对应的文件名
        transformed_images = []
        for transform_name, transform_func in transformations.items():
            # 创建变换后的图片
            transformed_img = transform_func(img)
            
            # 生成新的文件名
            new_file_name = f"{base_name}_{transform_name}.JPG"
            
            # 存储变换后的PIL Image对象和对应的文件名
            transformed_images.append((transformed_img, new_file_name))
            
        return transformed_images
    except Exception as e:
        print(f"变换图片时出错: {e}")
        return []

# 定义单个图像的处理函数
def process_single_image(image_file, input_folder, output_folder):
    try:
        input_path = os.path.join(input_folder, image_file)
        base_name = os.path.splitext(image_file)[0]  # 获取不含扩展名的文件名
        
        # 1. 预处理图像
        processed_img = process_image(input_path)
        if processed_img is None:
            print(f"   图像{image_file}预处理失败，跳过后续步骤")
            return False, 0
        
        # 2. 使用内存版本的函数分割图像
        split_images_with_labels = split(processed_img)
        if not split_images_with_labels:
            print(f"   图像{image_file}分割失败，跳过后续步骤")
            return False, 0
        
        # 3. 在内存中对所有分割后的图像应用变换操作
        all_images_to_save = []
        
        # 保存原始分割图的信息
        for cropped_img, region_label in split_images_with_labels:
            split_file_name = f"{base_name}_split_{region_label}.JPG"
            all_images_to_save.append((cropped_img, split_file_name))
            
            # 对每个分割图应用变换
            transform_base_name = f"{base_name}_split_{region_label}"
            transformed_images = transformations(cropped_img, transform_base_name)
            all_images_to_save.extend(transformed_images)
        
        # 4. 一次性保存所有处理好的图像到输出文件夹
        for img_to_save, file_name in all_images_to_save:
            output_path = os.path.join(output_folder, file_name)
            img_to_save.save(output_path)
        
        total_generated_files = len(all_images_to_save)
        return True, total_generated_files
    except Exception as e:
        print(f"处理图像{image_file}时发生错误: {e}")
        return False, 0

# 定义完整的图像处理流水线函数 - 多线程
def image_processing_pipeline(input_folder, output_folder, max_workers=None):
    try:
        # 确保输出文件夹存在
        os.makedirs(output_folder, exist_ok=True)
        
        # 获取输入文件夹中的所有JPG文件
        image_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.jpg')]
        
        if not image_files:
            print(f"错误：在{input_folder}中未找到JPG图片文件")
            return False
        
        total_images = len(image_files)
        print(f"找到{total_images}张图片待处理")
        print(f"使用多线程处理，最大工作线程数: {max_workers or '自动'}")
        
        # 统计信息
        processed_count = 0
        total_generated_files = 0
        
        # 使用线程池并行处理图像
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_image = {
                executor.submit(process_single_image, image_file, input_folder, output_folder): image_file 
                for image_file in image_files
            }
            
            # 使用tqdm显示进度条
            for future in tqdm(concurrent.futures.as_completed(future_to_image), total=total_images, desc="处理进度"):
                image_file = future_to_image[future]
                try:
                    success, generated_files = future.result()
                    if success:
                        processed_count += 1
                        total_generated_files += generated_files
                except Exception as e:
                    print(f"处理图像{image_file}时发生意外错误: {e}")
        
        print(f"\n图像处理流水线执行完成！")
        print(f"成功处理了{processed_count}张图片")
        print(f"总共生成了{total_generated_files}个文件")
        print(f"所有结果已保存到: {output_folder}")
        return True
    except Exception as e:
        print(f"执行图像处理流水线时发生错误: {e}")
        return False

# 主函数
if __name__ == "__main__":


    # 定义各文件夹路径
    # INPUT_FOLDER = "D:/素材/2024流星雨/JPG/轨迹/"
    INPUT_FOLDER = "D:/素材/2024流星雨/JPG/无轨迹/"
    # OUTPUT_FOLDER = "./DataSet/Positive/"
    OUTPUT_FOLDER = "./DataSet/Negative/"
    

    import multiprocessing
    cpu_count = multiprocessing.cpu_count()
    MAX_WORKERS = min(16, cpu_count * 2)  # 设置上限为16，避免创建过多线程
    
    print("=====================================================")
    print("                  图像处理流水线启动                   ")
    print("=====================================================")
    print(f"输入文件夹: {INPUT_FOLDER}")
    print(f"输出文件夹: {OUTPUT_FOLDER}")
    print(f"系统CPU核心数: {cpu_count}")
    print(f"设置的最大线程数: {MAX_WORKERS}")
    
    # 执行图像处理流水线
    success = image_processing_pipeline(
        INPUT_FOLDER,
        OUTPUT_FOLDER,
        max_workers=MAX_WORKERS
    )
    
    if success:
        print("\n图像处理流水线成功完成！")
    else:
        print("\n图像处理流水线执行失败，请查看错误信息。")