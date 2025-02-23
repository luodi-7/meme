import json

# 定义输入和输出文件路径
input_file = "/mnt/afs/xueyingyi/COCO/train_process/train_data.jsonl"
output_file_one = "/mnt/afs/xueyingyi/COCO/train_process/filtered_one.jsonl"
output_file_two = "/mnt/afs/xueyingyi/COCO/train_process/filtered_other.jsonl"

# 打开输入文件和输出文件
with open(input_file, "r") as infile, \
     open(output_file_one, "w") as outfile_one, \
     open(output_file_two, "w") as outfile_two:
    
    # 逐行读取jsonl文件
    for line in infile:
        item = json.loads(line.strip())
        
        # 标记是否符合筛选条件
        is_filtered = False
        
        # 检查每个对话的gpt回复
        for conversation in item["conversations"]:
            if conversation["from"] == "gpt":
                # 分割value字符串，检查列表长度是否为1
                if len(conversation["value"].split(',')) == 1:
                    is_filtered = True
                    break
        
        # 根据是否符合条件将数据写入不同的文件
        if is_filtered:
            json.dump(item, outfile_one)
            outfile_one.write("\n")
        else:
            json.dump(item, outfile_two)
            outfile_two.write("\n")

print("筛选完成！")
