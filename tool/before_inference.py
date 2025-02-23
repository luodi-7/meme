import os
import shutil

def move_files(source_dir, target_dir, file_names):
    """
    将指定文件从源目录移动到目标目录
    :param source_dir: 源目录路径
    :param target_dir: 目标目录路径
    :param file_names: 需要移动的文件名列表
    """
    # 确保目标目录存在
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"创建目标目录: {target_dir}")

    # 遍历文件名列表，移动文件
    for file_name in file_names:
        source_path = os.path.join(source_dir, file_name)
        target_path = os.path.join(target_dir, file_name)

        if os.path.exists(source_path):
            shutil.move(source_path, target_path)
            print(f"移动文件: {file_name} -> {target_path}")
        else:
            print(f"文件不存在: {source_path}")

if __name__ == "__main__":
    # 源目录路径
    source_dir = "/mnt/afs/xueyingyi/model/ch_meme_v6"

    # 目标目录路径
    target_dir = "/mnt/afs/xueyingyi/model/generate_few_shot_train"

    # 需要移动的文件名列表
    file_names = [
        "configuration_intern_vit.py",
        "configuration_internlm2.py",
        "configuration_internvl_chat.py",
        "conversation.py",
        "modeling_intern_vit.py",
        "modeling_internlm2.py",
        "modeling_internvl_chat.py"
    ]

    # 移动文件
    move_files(source_dir, target_dir, file_names)