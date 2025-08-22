import os
import json

# 配置参数
PARTS_DIR = './model_parts'
OUTPUT_FILE = './model.pth'

# 读取元数据
with open(os.path.join(PARTS_DIR, 'metadata.json'), 'r') as f:
    metadata = json.load(f)

num_parts = metadata['num_parts']

print(f"开始合并 {num_parts} 个模型部分...")

# 合并文件
with open(OUTPUT_FILE, 'wb') as out_f:
    for i in range(num_parts):
        part_file = os.path.join(PARTS_DIR, f'part_{i:03d}.bin')
        if not os.path.exists(part_file):
            print(f"错误: 找不到部分文件 {part_file}")
            exit(1)
        
        # 读取部分文件并写入输出文件
        with open(part_file, 'rb') as part_f:
            out_f.write(part_f.read())
        print(f"已合并部分 {i+1}/{num_parts}")

print(f"模型文件合并完成！")
print(f"重建的模型文件保存在: {OUTPUT_FILE}")

# 验证文件大小
reconstructed_size = os.path.getsize(OUTPUT_FILE)
original_size = metadata['original_size']
print(f"原始文件大小: {original_size} 字节")
print(f"重建文件大小: {reconstructed_size} 字节")

if reconstructed_size == original_size:
    print("✓ 文件大小匹配，合并成功！")
else:

    print("✗ 文件大小不匹配，合并可能有问题。")
