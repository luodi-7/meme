import json
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

# 定义输入和输出文件路径
jsonl_file = '/mnt/afs/xueyingyi/meme/data/Cjson/C_generate_train.jsonl'  # 输入 JSONL 文件
output_jsonl_file = '/mnt/afs/xueyingyi/meme/generate/C_generate_train_output.jsonl'  # 输出 JSONL 文件

# 定义图像预处理函数
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # 计算图像的宽高比
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # 找到最接近的目标宽高比
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # 计算目标宽度和高度
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # 调整图像大小
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # 裁剪图像
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

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

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    print(f"Processed {len(images)} blocks for image {image_file}")
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

# 加载模型和分词器
path = '/mnt/afs/xueyingyi/model/generate_text_v1'
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

# 打开输出文件
with open(output_jsonl_file, 'w') as output_file:
    # 读取 JSONL 文件
    with open(jsonl_file, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            image_path = data['image']  # 获取图片路径
            conversations = data['conversations']
            
            # 提取 human 部分的 value 作为提示词
            prompt = None
            gpt_value = None
            for conv in conversations:
                if conv['from'] == 'human':
                    prompt = conv['value']
                elif conv['from'] == 'gpt':
                    gpt_value = conv['value']  # 提取原始的 gpt value
            
            if not prompt or not gpt_value:
                print(f"Error: Missing human prompt or gpt value for image {image_path}")
                continue
            
            # 加载并预处理图像
            try:
                pixel_values = load_image(image_path, max_num=12).to(torch.bfloat16).cuda()
                assert pixel_values.numel() > 0, "Pixel values are empty!"
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                continue
            
            # 设置生成配置
            generation_config = dict(max_new_tokens=1024, do_sample=False, num_beams=1)
            
            # 使用提取的提示词进行推理
            try:
                response = model.chat(tokenizer, pixel_values, prompt, generation_config)
                print(f'Image: {image_path}\nPrompt: {prompt}\nGPT Value: {gpt_value}\nInference: {response}\n')
                
                # 构建输出数据
                output_data = {
                    "id": data["id"],  # 保留原始 ID
                    "image": image_path,  # 图片路径
                    "conversations": [
                        {"from": "gpt", "value": gpt_value},  # 原始的 gpt value
                        {"from": "inference", "value": response}  # 模型生成的文本
                    ]
                }
                # 写入输出文件
                output_file.write(json.dumps(output_data) + '\n')
            except RuntimeError as e:
                print(f"Error processing image {image_path}: {e}")