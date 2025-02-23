import os
import shutil

# 源路径
source_path = '/mnt/afs/xueyingyi/image_vague/mask_add'
# 目标路径
target_path = '/mnt/afs/xueyingyi/image_vague/mask_demo'

# 确保目标路径存在
os.makedirs(target_path, exist_ok=True)

# 遍历源路径下的所有文件
for filename in os.listdir(source_path):
    # 检查文件名是否符合条件
    if filename.startswith('image_ (') and filename.endswith(').jpg'):
        # 构建源文件和目标文件的完整路径
        source_file = os.path.join(source_path, filename)
        target_file = os.path.join(target_path, filename)
        
        # 复制文件
        shutil.copy2(source_file, target_file)
        print(f'Copied {source_file} to {target_file}')