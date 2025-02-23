import csv
import json
import argparse
import numpy as np
import random

# change the sequence of json dict to augment the dataset
# First metaphor, then offensive，then emotion
SEQ_AUG = ['emo_off_meta']# , 'meta_off_emo', 'meta_emo_off']

def load_data(path):
    with open(path, 'r', encoding = "utf-8") as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        data = [row for row in reader]

    random.shuffle(data)

    # 取前90%的元素
    selected_items = data[:int(len(data) * 0.9)]
    unselected_items = data[int(len(data) * 0.9):]

    with open('/mnt/afs/xueyingyi/meme/data/label_C_train.csv', 'w') as file:
        writer = csv.writer(file)
        for item in selected_items:
            writer.writerow(item)

    with open('/mnt/afs/xueyingyi/meme/data/label_C_evaluate.csv', 'w') as file:
        writer = csv.writer(file)
        for item in unselected_items:
            writer.writerow(item)
    
    return selected_items, unselected_items

def load_csv_data(path):
    with open(path, 'r', encoding = "utf-8") as file:
        reader = csv.reader(file)
        data = [row for row in reader]
    return data

def build_input(image_path, json_path_list, output_path, length_list):
    dict_list = []
    for json_path, length in zip(json_path_list, length_list):
        dict = {
            "classification_C": {
            "root": image_path,
            "annotation": json_path,
            "data_augment": False,
            "repeat_time": 1,
            "length": length
            }
        }
        dict_list.append(dict)

    with open(output_path, 'w', encoding='utf-8') as file:
        for dict in dict_list:
            # 使用 json.dumps() 将字典转换为 JSON 格式的字符串
            json.dump(dict, file)
            # 写入换行符，以便每个字典占据文件中的一行
            file.write('\n')

def build_json(data, json_path, seq, relabel_path=None):
    # build data from label_C
    # load prompt from file
    with open(f'/mnt/afs/niuyazhe/data/meme/prompt_classification_{seq}.txt', 'r') as file:
        PROMPT = file.read()
    
    
    image_path = '/mnt/afs/niuyazhe/data/meme/data/Cimages/Cimages/Cimages/'

    dict_list = []

    for id, d in enumerate(data):
        name, senti_cate, senti_deg, intent, offense, meta_occur, meta_cate, target, source, target_mod, source_mod = d
        if relabel_path:
            flag = False
            with open(relabel_path, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    if row[0] == name:
                        if 'character' not in row[1] or 'reader' not in row[2] and 'author' not in row[2]:
                            flag = True
                        break
            if flag:
                continue

        if seq == 'emo_off_meta':
            answer_dict = {
                'sentiment_category': senti_cate[2:-1], # replace_with_zh(senti_cate[2:-1], False)
                'sentiment_degree': senti_deg[2:-1],
                'intention_detection': intent[2:-1], # replace_with_zh(intent[2:-1], False),
                'offensiveness_detection': offense[2:-1],
                'metaphor_occurrence': meta_occur,
                'metaphor_category': meta_cate,
                'target_domain': target,
                'source_domain': source,
                'target_modality': target_mod,
                'source_modality': source_mod
                }
        elif seq == 'meta_off_emo':
            answer_dict = {
                'metaphor_occurrence': meta_occur,
                'metaphor_category': meta_cate,
                'target_domain': target,
                'source_domain': source,
                'target_modality': target_mod,
                'source_modality': source_mod,
                'intention_detection': intent[2:-1],
                'offensiveness_detection': offense[2:-1],
                'sentiment_category': senti_cate[2:-1],
                'sentiment_degree': senti_deg[2:-1]
            }
        elif seq == 'meta_emo_off':
            answer_dict = {
                'metaphor_occurrence': meta_occur,
                'metaphor_category': meta_cate,
                'target_domain': target,
                'source_domain': source,
                'target_modality': target_mod,
                'source_modality': source_mod,
                'sentiment_category': senti_cate[2:-1],
                'sentiment_degree': senti_deg[2:-1],
                'intention_detection': intent[2:-1],
                'offensiveness_detection': offense[2:-1],
            }
        else:
            raise ValueError('wrong sequence')
        
        data_json = {'id': id,
                    'image': image_path+name,
                    'conversations': [
                        {'from': 'human', 'value': f'<image>{PROMPT}'}, # f'<image>{replace_with_zh(PROMPT, True)}
                        {'from': 'gpt', 'value': f'{answer_dict}'}
                    ]}

        dict_list.append(data_json)
    
    with open(json_path, 'w', encoding='utf-8') as file:
        for entry in dict_list:
            # 使用 json.dumps() 将字典转换为 JSON 格式的字符串
            json.dump(entry, file)
            # 写入换行符，以便每个字典占据文件中的一行
            file.write('\n')

    return image_path, len(dict_list)

def replace_with_zh(text, prompt = True):
    text = text.replace('happiness', '幸福')
    text = text.replace('love', '爱')
    text = text.replace('anger', '愤怒')
    text = text.replace('sorrow', '悲伤')
    text = text.replace('fear', '恐惧')
    text = text.replace('hate', '憎恨')
    text = text.replace('surprise', '惊讶')

    text = text.replace('interactive', '互动')
    text = text.replace('expressive', '表达')
    text = text.replace('entertaining', '有趣')
    if prompt:
        text = text.replace('/offensive/', '/冒犯/')
        text = text.replace('\'offensive\'', '\'冒犯\'')
        text = text.replace('/other', '/其他')
    else:
        text = text.replace('offensive', '冒犯')
        text = text.replace('other', '其他')

    return text

def build_cot_json(data, json_path, seq, relabel_path=None):
    # build data from label_E
    # load prompt from file
    with open(f'/mnt/afs/niuyazhe/data/meme/prompt_classification_{seq}_cot.txt', 'r') as file:
        PROMPT = file.read()
    
    image_path = '/mnt/afs/niuyazhe/data/meme/data/Cimages/Cimages/Cimages/'

    dict_list = []

    count = 0
    for id, d in enumerate(data):
        name, senti_cate, senti_deg, intent, offense, meta_occur, meta_cate, target, source, target_mod, source_mod = d

        if relabel_path:
            flag = False
            with open(relabel_path, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    if row[0] == name:
                        if 'character' not in row[1] or 'reader' not in row[2] and 'author' not in row[2]:
                            flag = True
                        break
            if flag:
                continue
        for i in range(4):
            if i > 0:
                new_name = name[:-4] + f'_{i-1}' + name[-4:]
            else:
                new_name = name
            if seq == 'emo_off_meta':
                cot_dict = {                
                    'metaphor_occurrence': meta_occur,
                    'metaphor_category': meta_cate,
                    'target_domain': target,
                    'source_domain': source,
                    'target_modality': target_mod,
                    'source_modality': source_mod
                    }
                answer_dict = {
                    'sentiment_category': senti_cate[2:-1],
                    'sentiment_degree': senti_deg[2:-1],
                    'intention_detection': intent[2:-1],
                    'offensiveness_detection': offense[2:-1],
                    }
            else:
                raise ValueError('wrong sequence')
            
            data_json = {'id': count,
                        'image': image_path+new_name,
                        'conversations': [
                            {'from': 'human', 'value': f'<image>{PROMPT}\nThe metophor in the sequence is {cot_dict}'},
                            {'from': 'gpt', 'value': f'{answer_dict}'}
                        ]}
            dict_list.append(data_json)
            count += 1
    
    with open(json_path, 'w', encoding='utf-8') as file:
        for entry in dict_list:
            # 使用 json.dumps() 将字典转换为 JSON 格式的字符串
            json.dump(entry, file)
            # 写入换行符，以便每个字典占据文件中的一行
            file.write('\n')

    return image_path, len(dict_list)
    

def build_classification(data, json_path):
    with open('/mnt/afs/xueyingyi/meme/prompt_classification_emo_off_meta.txt', 'r') as file:
        PROMPT = file.read()

    image_path = '/mnt/afs/niuyazhe/data/meme/data/Cimages/Cimages/Cimages/'

    dict_list = []

    sentiment_categories = ['happiness', 'anger', 'sorrow', 'fear', 'hate', 'surprise']
    sentiment_degrees = ['slightly', 'moderate', 'very']
    intention_categories = ['interactive', 'expressive', 'entertaining', 'offensive', 'other']
    offensiveness_categories = ['non-offensive', 'slightly', 'moderate', 'very']
    metaphor_categories = ['image dominant', 'text dominant', 'complementary']
    target_modality_categories = ['image', 'text', 'complementary']
    source_modality_categories = ['image', 'text', 'complementary']

    key_list = ['sentiment_category', 'sentiment_degree', 'intention_detection',
        'offensiveness_detection', 'metaphor_occurrence', 'metaphor_category',
        'target_modality', 'source_modality']
    num_labels_list = [7, 3, 5, 4, 2, 4, 4, 4]

    for id, d in enumerate(data):
        name, senti_cate, senti_deg, intent, offense, meta_occur, meta_cate, target, source, target_mod, source_mod = d
        answer_dict = {
            'sentiment_category': senti_cate[0],
            'sentiment_degree': senti_deg[0],
            'intention_detection': intent[0],
            'offensiveness_detection': offense[0],
            'metaphor_occurrence': meta_occur,
            'metaphor_category': meta_cate,
            'target_modality': target_mod,
            'source_modality': source_mod
            }

        label_list = []

        for i, key in enumerate(key_list):
            label = [0]*num_labels_list[i]
            if i < 3:
                label[int(answer_dict[key])-1]=1
            if key == 'offensiveness_detection':
                label[int(answer_dict[key])]=1
            else:
                if key == 'metaphor_occurrence':
                    label[int(answer_dict[key])] = 1
                elif key == 'metaphor_category':
                    if 'image' in answer_dict[key]:
                        label[0]=1
                    elif 'text' in answer_dict[key]:
                        label[1]=1
                    elif 'complementary' in answer_dict[key]:
                        label[2]=1
                    else:
                        label[3]=1
                elif key == 'target_modality':
                    if 'image' in answer_dict[key]:
                        label[0]=1
                    elif 'text' in answer_dict[key]:
                        label[1]=1
                    elif 'complementary' in answer_dict[key]:
                        label[2]=1
                    else:
                        label[3]=1
                elif key == 'source_modality':
                    if 'image' in answer_dict[key]:
                        label[0]=1
                    elif 'text' in answer_dict[key]:
                        label[1]=1
                    elif 'complementary' in answer_dict[key]:
                        label[2]=1
                    else:
                        label[3]=1
            label_list.append(label)

        data_json = {'id': id,
                    'image': image_path + name,
                    'conversations': [
                        {'from': 'human', 'value': f'<image>'},
                        {'from': 'gpt', 'value': label_list}
                    ]}
        dict_list.append(data_json)

    with open(json_path, 'w', encoding='utf-8') as file:
        for entry in dict_list:
            json.dump(entry, file)
            file.write('\n')

    return image_path, json_path, len(dict_list)


        
    
    with open(json_path, 'w', encoding='utf-8') as file:
        for entry in dict_list:
            # 使用 json.dumps() 将字典转换为 JSON 格式的字符串
            json.dump(entry, file)
            # 写入换行符，以便每个字典占据文件中的一行
            file.write('\n')

    return image_path, json_path, len(dict_list)

# def clean_generated_text(text):  
#     # 可以添加正则表达式去掉非中文字符  
#     import re  
#     cleaned_text = re.sub(r'[^\u4e00-\u9fa5]', '', text)  
#     return cleaned_text
    
if __name__ == '__main__':
    # input some parameters with argparse
    parser = argparse.ArgumentParser()
    # data_path='/mnt/afs/niuyazhe/data/meme/data/label_C.csv'
    # load_data(data_path)

    train_data_path = '/mnt/afs/xueyingyi/meme/data/label_C_train.csv'
    eval_data_path = '/mnt/afs/xueyingyi/meme/data/label_C_evaluate.csv'
    type = 'json' # or 'classification' or 'json' or 'cot'

    train_data = load_csv_data(train_data_path)
    eval_data = load_csv_data(eval_data_path)
    if type == 'json':
        json_path_train_list = []
        json_path_eval_list = []
        length_train_list = []
        length_eval_list = []
        for seq in SEQ_AUG:
            json_path_train = f'/mnt/afs/xueyingyi/meme/data/Cjson/Cjson_{seq}_relabel.jsonl'
            json_path_eval = f'/mnt/afs/xueyingyi/meme/data/Cjson/Cjson_eval_{seq}_relabel.jsonl'
            
            image_path_train, length_train = build_json(train_data, json_path_train, seq,'/mnt/afs/xueyingyi/meme/data/label_C_train_relabel.csv')
            image_path_eval, length_eval = build_json(eval_data, json_path_eval, seq,'/mnt/afs/xueyingyi/meme/data/label_C_evaluate_relabel.csv')

            json_path_train_list.append(json_path_train)
            json_path_eval_list.append(json_path_eval)
            length_train_list.append(length_train)
            length_eval_list.append(length_eval)

        output_path_train = f'/mnt/afs/xueyingyi/meme/data/data_C_json_relabel.jsonl'
        output_path_eval = f'/mnt/afs/xueyingyi/meme/data/data_C_json_eval_relabel.jsonl'
        build_input(image_path_train, json_path_train_list, output_path_train, length_train_list)
        build_input(image_path_eval, json_path_eval_list, output_path_eval, length_eval_list)
    elif type == 'cot':
        seq = 'emo_off_meta'
        json_path_train = f'/mnt/afs/xueyingyi/meme/data/Cjson/Cjson_{seq}_cot_with_zh_relabel.jsonl'
        json_path_eval = f'//mnt/afs/xueyingyi/meme/data/Cjson/Cjson_eval_{seq}_cot_with_zh_relabel.jsonl'

        image_path_train, length_train = build_cot_json(train_data, json_path_train, seq,'/mnt/afs/xueyingyi/meme/data/label_C_train_relabel.csv')
        image_path_eval, length_eval = build_cot_json(eval_data, json_path_eval, seq,'/mnt/afs/xueyingyi/meme/data/label_C_evaluate_relabel.csv')

        output_path_train = f'/mnt/afs/xueyingyi/meme/data/data_C_json_cot_with_zh_relabel.jsonl'
        output_path_eval = f'/mnt/afs/xueyingyi/meme/data/data_C_json_eval_cot_with_zh_relabel.jsonl'
        build_input(image_path_train, [json_path_train], output_path_train, [length_train])
        build_input(image_path_eval, [json_path_eval], output_path_eval, [length_eval])

    else:
        # Generate classification jsonl files
        json_path_train = '/mnt/afs/xueyingyi/meme/data/Cjson/Cjson_classification.jsonl'
        json_path_eval = '/mnt/afs/xueyingyi/meme/data/Cjson/Cjson_classification_eval.jsonl'
        
        # Generate train and eval JSON files
        image_path_train, json_path_train, length_train = build_classification(train_data, json_path_train)
        image_path_eval, json_path_eval, length_eval = build_classification(eval_data, json_path_eval)
        
        # Generate input files for fine-tuning
        output_path_train = '/mnt/afs/xueyingyi/meme/data/data_C_classification.jsonl'
        output_path_eval = '/mnt/afs/xueyingyi/meme/data/data_C_classification_eval.jsonl'
        
        build_input(image_path_train, [json_path_train], output_path_train, [length_train])
        build_input(image_path_eval, [json_path_eval], output_path_eval, [length_eval])

    # image_path_train, length_train = build_json(train_data, json_path_train)
    # image_path_eval, length_eval = build_json(eval_data, json_path_eval)
    # build_input(image_path_train, json_path_train, output_path_train, length_train)
    # build_input(image_path_eval, json_path_eval, output_path_eval, length_eval)