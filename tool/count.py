import os

# 定义路径
path = '/mnt/afs/xueyingyi/image_vague/quickmeme_inpainting'

# 获取路径下的所有文件
files = os.listdir(path)

# 过滤出文件（排除文件夹）
file_count = len([f for f in files if os.path.isfile(os.path.join(path, f))])

# 输出文件数量
print(f"文件数量: {file_count}")