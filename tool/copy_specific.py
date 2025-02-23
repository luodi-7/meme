import os
import shutil

def copy_images_from_list(image_list, target_dir, replace_from, replace_to):
    # 确保目标目录存在，如果不存在则创建
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # 遍历图片路径列表
    for image_path in image_list:
        # 替换路径中的指定部分
        new_image_path = image_path.replace(replace_from, replace_to)
        
        # 检查文件是否存在
        if os.path.isfile(new_image_path):
            # 获取文件名
            filename = os.path.basename(new_image_path)
            # 构建目标文件的完整路径
            target_file = os.path.join(target_dir, filename)
            
            # 复制文件
            shutil.copy2(new_image_path, target_file)
            print(f"Copied {new_image_path} to {target_file}")
        else:
            print(f"File not found: {new_image_path}")

# 示例用法
image_list = [
    "/mnt/afs/xueyingyi/image_vague/combined/image_ (664).jpg",
    "/mnt/afs/xueyingyi/image_vague/combined/image_ (55).jpg",
    "/mnt/afs/xueyingyi/image_vague/combined/image_ (71).jpg",
    "/mnt/afs/xueyingyi/image_vague/combined/image_ (3653).jpg",
    "/mnt/afs/xueyingyi/image_vague/combined/image_ (3580).jpg",
    "/mnt/afs/xueyingyi/image_vague/combined/image_ (3212).jpg",
    "/mnt/afs/xueyingyi/image_vague/combined/image_ (2893).jpg",
    "/mnt/afs/xueyingyi/image_vague/combined/image_ (2677).jpg",
    "/mnt/afs/xueyingyi/image_vague/combined/image_ (2437).jpg",
    "/mnt/afs/xueyingyi/image_vague/combined/image_ (1945).jpg",
    "/mnt/afs/xueyingyi/image_vague/combined/image_ (1226).jpg",
    "/mnt/afs/xueyingyi/image_vague/combined/image_ (1191).jpg"
]

# 目标目录
target_directory = "/mnt/afs/xueyingyi/image_vague/ocr_failure"

# 需要替换的路径部分
replace_from = "/mnt/afs/xueyingyi/image_vague/combined/"
replace_to = "/mnt/afs/xueyingyi/image_vague/image/"

# 调用函数
copy_images_from_list(image_list, target_directory, replace_from, replace_to)