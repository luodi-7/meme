import os
import json
import ast
from collections import defaultdict
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

# 定义常量
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SENTIMENT_CATEGORY = ['happiness', 'love', 'anger', 'sorrow', 'fear', 'hate', 'surprise']
INTENTION_DETECTION = ['interactive', 'expressive', 'entertaining', 'offensive']
OFFENSIVENESS_DETECTION = ['non-offensive', 'slightly', 'moderately', 'very']
METAPHOR_CATEGORY = ['image dominant', 'text dominant', 'complementary']
TARGET_MODALITY = ['image', 'text', 'complementary']
SOURCE_MODALITY = ['image', 'text', 'complementary']

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def calculate_accuracy(pred_list, true_list):
    """计算准确率"""
    correct = sum(p == t for p, t in zip(pred_list, true_list))
    return correct / len(pred_list) if len(pred_list) > 0 else 0

def main():
    # Step 1: 从 JSONL 文件中读取图像路径
    jsonl_path = '/mnt/afs/xueyingyi/meme/data/Cjson/Cjson_eval_emo_off_meta_relabel.jsonl'
    image_paths = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            image_paths.append(data['image'])

    # Step 2: 加载模型和分词器
    path = '/mnt/afs/xueyingyi/model/ch_meme_v6'
    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

    # Step 3: 第一阶段推理
    quickmeme = []
    for image_path in image_paths:
        try:
            # 加载图像并预处理
            pixel_values = load_image(image_path, max_num=12).to(torch.bfloat16).cuda()
            generation_config = dict(max_new_tokens=1024, do_sample=False)

            # 读取系统提示
            with open("/mnt/afs/xueyingyi/meme/prompt_classification_emo_off_meta.txt", 'r', encoding='utf-8') as f:
                system_prompt = f.read()
            prompt = f"<image>\n{system_prompt}"

            # 调用模型生成元数据
            meme_emotion = model.chat(tokenizer, pixel_values, prompt, generation_config)
            meme = ast.literal_eval(meme_emotion)
            meme["uid"] = image_path
            print(meme)
            quickmeme.append(meme)

            # 保存第一阶段结果
            with open('/mnt/afs/xueyingyi/meme/TwoStage_inference/ch_meme_label.json', 'w') as json_file:
                json.dump(quickmeme, json_file, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue

    #Step 4: 第二阶段推理
    updated_quickmeme = []
    for meme in quickmeme:
        try:
            image_path = meme["uid"]
            mataphor_dict = {
                'metaphor_occurrence': f"{meme['metaphor_occurrence']}",
                'metaphor_category': f"{meme['metaphor_category']}",
                'target_domain': f"{meme['target_domain']}",
                'source_domain': f"{meme['source_domain']}",
                'target_modality': f"{meme['target_modality']}",
                'source_modality': f"{meme['source_modality']}"
            }

            # 加载图像并预处理
            pixel_values = load_image(image_path, max_num=12).to(torch.bfloat16).cuda()
            generation_config = dict(max_new_tokens=1024, do_sample=False)

            # 读取系统提示
            with open("/mnt/afs/niuyazhe/data/meme/prompt_classification_emo_off_meta_cot.txt", 'r', encoding='utf-8') as f:
                system_prompt = f.read()
            prompt = f"<image>{system_prompt}\nThe metophor in the sequence is{mataphor_dict}"

            # 调用模型生成更新后的元数据
            updated_meme_emotion = model.chat(tokenizer, pixel_values, prompt, generation_config)
            emotion = ast.literal_eval(updated_meme_emotion)

            # 更新元数据
            for key in ['sentiment_category', 'sentiment_degree', 'intention_detection', 'offensiveness_detection']:
                meme[key] = emotion[key]
            print(meme)
            updated_quickmeme.append(meme)

            # 保存第二阶段结果
            with open('/mnt/afs/xueyingyi/meme/TwoStage_inference/ch_meme_update_label.json', 'w') as json_file:
                json.dump(updated_quickmeme, json_file, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"Error processing {meme}: {e}")
            continue

    # Step 5: 评估
    with open('/mnt/afs/xueyingyi/meme/TwoStage_inference/ch_meme_label.json', 'r', encoding='utf-8') as json_file:
        new_quickmeme = json.load(json_file)
    print(len(new_quickmeme))
    true_labels = {}
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            image_path = data['image']
            true_label = ast.literal_eval(data['conversations'][1]['value'])
            true_labels[image_path] = true_label

    acc_dict = defaultdict(int)
    sentiment_true_list = []
    sentiment_pred_list = []
    intention_true_list = []
    intention_pred_list = []
    offensiveness_true_list = []
    offensiveness_pred_list = []
    sentiment_dict = defaultdict(int)
    intention_dict = defaultdict(int)
    offensiveness_dict = defaultdict(int)
    sentiment_length_dict = defaultdict(int)
    intention_length_dict = defaultdict(int)
    offensiveness_length_dict = defaultdict(int)

    count = 0
    for pred in new_quickmeme:
        image_path = pred['uid']
        if image_path in true_labels:
            true_label = true_labels[image_path]
            for key in true_label:
                if key in pred:
                    if key == 'sentiment_category':
                        sentiment_true_list.append(SENTIMENT_CATEGORY.index(true_label[key]))
                        sentiment_pred_list.append(SENTIMENT_CATEGORY.index(pred[key]))
                        sentiment_length_dict[true_label[key]] += 1
                        if pred[key] == true_label[key]:
                            sentiment_dict[true_label[key]] += 1

                    elif key == 'intention_detection' and pred[key] in INTENTION_DETECTION:
                        intention_true_list.append(INTENTION_DETECTION.index(true_label[key]))
                        intention_pred_list.append(INTENTION_DETECTION.index(pred[key]))
                        intention_length_dict[true_label[key]] += 1
                        if pred[key] == true_label[key]:
                            intention_dict[true_label[key]] += 1

                    elif key == 'offensiveness_detection' and pred[key] in OFFENSIVENESS_DETECTION:
                        offensiveness_true_list.append(OFFENSIVENESS_DETECTION.index(true_label[key]))
                        offensiveness_pred_list.append(OFFENSIVENESS_DETECTION.index(pred[key]))
                        offensiveness_length_dict[true_label[key]] += 1
                        if pred[key] == true_label[key]:
                            offensiveness_dict[true_label[key]] += 1

                    if pred[key] == true_label[key]:
                        acc_dict[key] += 1
            count += 1

    for key in acc_dict:
        acc_dict[key] /= count

    for key in SENTIMENT_CATEGORY:
        if sentiment_length_dict[key] > 0:
            acc_dict[f'sentiment_{key}'] = sentiment_dict[key] / sentiment_length_dict[key]

    for key in INTENTION_DETECTION:
        if intention_length_dict[key] > 0:
            acc_dict[f'intention_{key}'] = intention_dict[key] / intention_length_dict[key]

    for key in OFFENSIVENESS_DETECTION:
        if offensiveness_length_dict[key] > 0:
            acc_dict[f'offensiveness_{key}'] = offensiveness_dict[key] / offensiveness_length_dict[key]

    print("Accuracy Results:")
    for key, value in acc_dict.items():
        print(f"{key}: {value:.4f}")

if __name__ == "__main__":
    main()