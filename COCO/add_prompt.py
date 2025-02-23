import json

# 读取 prompt.txt 文件内容
prompt_file_path = "/mnt/afs/xueyingyi/COCO/prompt.txt"
with open(prompt_file_path, "r", encoding="utf-8") as f:
    prompt_content = f.read().strip()  # 读取内容并去除首尾空白字符

# 读取 test_data.jsonl 文件并处理
input_file_path = "/mnt/afs/xueyingyi/COCO/train_process/filtered_one.jsonl"
output_file_path = "/mnt/afs/xueyingyi/COCO/train_process/filtered_one_prompt_add.jsonl"

with open(input_file_path, "r", encoding="utf-8") as infile, open(output_file_path, "w", encoding="utf-8") as outfile:
    for line in infile:
        # 解析 JSON 行
        data = json.loads(line.strip())
        
        # 找到 human 部分的 value
        for conversation in data["conversations"]:
            if conversation["from"] == "human":
                # 在原有 value 前添加 prompt 内容
                conversation["value"] = f"{prompt_content} {conversation['value']}"
                break  # 找到后跳出循环
        
        # 将处理后的数据写入新文件
        outfile.write(json.dumps(data, ensure_ascii=False) + "\n")

print(f"处理完成，结果已保存到 {output_file_path}")