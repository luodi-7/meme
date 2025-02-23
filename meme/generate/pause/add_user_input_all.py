import json
import random
from copy import deepcopy

def assign_pause_tokens_to_keys(keys, pause_token_count):
    """
    为每个键预分配一组pause token
    :param keys: 需要分配pause token的键列表
    :param pause_token_count: 每个键分配几个pause token
    :return: 一个字典，键为user_input的键，值为对应的pause token列表
    """
    pause_token_map = {}
    pause_counter = 0
    for key in keys:
        pause_tokens = [f"<pause_{pause_counter + j}>" for j in range(pause_token_count)]
        pause_token_map[key] = " ".join(pause_tokens)
        pause_counter += pause_token_count
    return pause_token_map

def mask_user_input(user_input, max_mask_count, pause_token_map):
    """
    随机对user_input中的某些键进行mask，使用预分配的pause token
    :param user_input: 原始的user_input字典
    :param max_mask_count: 最多覆盖几个键的值
    :param pause_token_map: 预分配的pause token字典
    :return: 处理后的user_input字典
    """
    keys = list(pause_token_map.keys())
    
    # 随机选择要mask的键
    mask_count = random.randint(1, max_mask_count)
    selected_keys = random.sample(keys, mask_count)
    
    # 对选中的键进行mask，使用预分配的pause token
    for key in selected_keys:
        user_input[key] = pause_token_map[key]
    
    return user_input

def generate_masked_data(data, max_mask_count, pause_token_count, mask_keys=None):
    """
    生成mask后的数据
    :param data: 原始数据列表
    :param max_mask_count: 最多覆盖几个键的值
    :param pause_token_count: 每个被覆盖的键值插入几个pause token
    :param mask_keys: 指定要覆盖的键列表，如果为None，则使用所有键
    :return: mask后的数据列表
    """
    # 确定需要分配pause token的键
    if mask_keys is None:
        mask_keys = list(data[0]['user_input'].keys())  # 默认使用所有键
    
    # 为每个键预分配pause token
    pause_token_map = assign_pause_tokens_to_keys(mask_keys, pause_token_count)
    
    # 生成mask后的数据
    masked_data = []
    for item in data:
        masked_item = deepcopy(item)
        masked_item['user_input'] = mask_user_input(masked_item['user_input'], max_mask_count, pause_token_map)
        masked_data.append(masked_item)
    return masked_data

def split_data(data, train_ratio=0.9):
    """
    划分训练集和测试集
    :param data: 数据列表
    :param train_ratio: 训练集比例
    :return: 训练集和测试集
    """
    random.shuffle(data)
    train_size = int(len(data) * train_ratio)
    train_data = data[:train_size]
    eval_data = data[train_size:]
    return train_data, eval_data

def save_jsonl(data, file_path):
    """
    保存数据为JSONL文件
    :param data: 数据列表
    :param file_path: 文件路径
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def main():
    # 文件路径
    input_jsonl_path = '/mnt/afs/xueyingyi/meme/generate/user_input_all.jsonl'  # 输入JSONL文件路径
    output_jsonl_path = '/mnt/afs/xueyingyi/meme/generate/omit/C_generate_multi.jsonl'  # 训练集路径


    # 读取user_input_all.jsonl文件
    data = []
    with open(input_jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))

    # 参数设置
    max_mask_count = 2  # 最多覆盖2个键的值
    pause_token_count = 3  # 每个被覆盖的键值插入3个pause token
    mask_keys = ["Emotion Category", "Emotion Intensity", "Intention Category", "Scene or Theme", "Style Preference", "Text Content Keywords"]  # 指定要覆盖的键

    # 生成mask后的数据
    masked_data = generate_masked_data(data, max_mask_count, pause_token_count, mask_keys)



    # 保存训练集和测试集
    save_jsonl(masked_data, output_jsonl_path)

    print("数据处理完成！")
    print(f"训练集大小: {len(masked_data)}")


if __name__ == "__main__":
    main()