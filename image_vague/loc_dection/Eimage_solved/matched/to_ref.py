import json
import re

# 读取原始的JSONL文件
input_file = '/mnt/afs/xueyingyi/image_vague/loc_dection/Eimage_solved/matched/eval_data_format.jsonl'
output_file = '/mnt/afs/xueyingyi/image_vague/loc_dection/Eimage_solved/matched/eval_data_ref.jsonl'

# 定义一个函数来去除坐标中的无效0
def remove_leading_zeros(coordinate):
    return [int(num) for num in coordinate]

# 定义一个函数来修改每一行
def modify_line(line):
    data = json.loads(line)

    # 修改image字段为列表，第一项是image_ (0).jpg，第二项是原来的图片
    image_path = data['image']
    image_name = image_path.split('/')[-1]
    new_image_path = image_path.replace(image_name, f"image_ (0).jpg")
    data['image'] = [new_image_path, image_path]

    # 修改human的value，插入示例图片和文本
    human_value = data['conversations'][0]['value']
    new_human_value = human_value.replace(
        "<image>\n Now", 
        "\nFor example, this picture:\n<image>\nthis sentence:That moment after you throw up and your friend asks you \"YOU GOOD BRO?\" I'M FUCKIN LIT\nthe answer should be:<ref>Writable text area</ref><box>[[0000, 0001, 0992, 0290]]</box>:that moment after you throw up and your friend asks you \"you good bro?\",\n<ref>Writable text area</ref><box>[[0276, 0801, 0746, 0903]]</box>:i'm fuckin lit\nNow look at this picture:\n<image>\n Now"
    )
    data['conversations'][0]['value'] = new_human_value

    # 修改gpt的value，在每个<box>前加<ref>Writable text area</ref>
    gpt_value = data['conversations'][1]['value']
    new_gpt_value = gpt_value.replace(
        "<box>", 
        "<ref>Writable text area</ref><box>"
    )
    

    data['conversations'][1]['value'] = new_gpt_value

    return json.dumps(data)

# 读取原始文件并写入新的文件
with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    for line in infile:
        modified_line = modify_line(line)
        outfile.write(modified_line + '\n')

print(f"文件处理完成，已保存为 {output_file}")
