import os
import re

# 设置目标目录
root_dir = '/mnt/afs/niuyazhe/data/meme/data/Cimages/Cimages/Cimages/'

# 遍历目录中的所有文件
for filename in os.listdir(root_dir):
    old_path = os.path.join(root_dir, filename)
    
    # 检查文件是否为jpg文件
    if os.path.isfile(old_path) and filename.endswith('.jpg'):
        
        # 使用正则表达式匹配文件名中的 "Image- (数字).jpg"
        match = re.match(r"Image-\s?\((\d+)\)\.jpg", filename)
        
        if match:
            # 获取数字部分
            num = match.group(1)
            
            # 生成新的文件名，替换 "-" 为 "_"
            new_filename = f"Image_({num}).jpg"
            new_path = os.path.join(root_dir, new_filename)
            
            # 重命名文件
            os.rename(old_path, new_path)
            print(f'Renamed: {old_path} -> {new_path}')
