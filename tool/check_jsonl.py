# 文件路径
jsonl_file_path = '/mnt/afs/xueyingyi/meme/generate/quickmeme/user_input_descriptions_simple.jsonl'

# 初始化变量
line_number = 0  # 当前行号
first_invalid_line = None  # 第一个不符合条件的行号

# 打开文件并逐行检查
with open(jsonl_file_path, 'r') as file:
    for line in file:
        line_number += 1
        # 检查是否是 13 的倍数行
        if line_number % 13 == 0:
            # 检查是否以 "```" 开头
            if not line.strip().startswith("```"):
                first_invalid_line = line_number
                break  # 找到第一个不符合条件的行后退出

# 输出结果
if first_invalid_line:
    print(f"第一个不符合条件的行号是: {first_invalid_line}")
else:
    print("所有 13 的倍数行均以 '```' 开头。")