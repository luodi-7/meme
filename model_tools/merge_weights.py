import argparse
import torch
from safetensors import safe_open
from internvl.model.internvl_chat import InternVLChatModel
from transformers import AutoTokenizer

def merge_weights_and_log(model, saved_weights, log_file):
    """
    合并权重并记录合并结果
    :param model: 模型对象
    :param saved_weights: 加载的权重字典
    :param log_file: 日志文件路径
    """
    with open(log_file, 'w') as f:
        # 合并 lm_head 权重
        if 'language_model.base_model.model.lm_head.new_linear.weight' in saved_weights and \
           'language_model.base_model.model.lm_head.original_linear.weight' in saved_weights:
            new_weight = saved_weights['language_model.base_model.model.lm_head.new_linear.weight']
            original_weight = saved_weights['language_model.base_model.model.lm_head.original_linear.weight']
            merged_weight = torch.cat([original_weight, new_weight], dim=0)
            
            model.language_model.lm_head.weight.data = merged_weight
            f.write("'language_model.lm_head.weight' 已 merge\n")
        else:
            f.write("'language_model.lm_head.weight' 未 merge（缺少权重）\n")

        # 合并 embed_tokens 权重
        if 'language_model.base_model.model.model.embed_tokens.new_embedding.weight' in saved_weights and \
           'language_model.base_model.model.model.embed_tokens.original_embedding.weight' in saved_weights:
            new_embedding = saved_weights['language_model.base_model.model.model.embed_tokens.new_embedding.weight']
            original_embedding = saved_weights['language_model.base_model.model.model.embed_tokens.original_embedding.weight']
            merged_embedding = torch.cat([original_embedding, new_embedding], dim=0)
            model.language_model.model.model.embed_tokens.weight.data = merged_embedding
            f.write("'language_model.model.model.embed_tokens.weight' 已 merge\n")
        else:
            f.write("'language_model.model.model.embed_tokens.weight' 未 merge（缺少权重）\n")

        # 检查其他权重是否被 merge
        for key in saved_weights.keys():
            if key not in [
                'language_model.base_model.model.lm_head.new_linear.weight',
                'language_model.base_model.model.lm_head.original_linear.weight',
                'language_model.base_model.model.model.embed_tokens.new_embedding.weight',
                'language_model.base_model.model.model.embed_tokens.original_embedding.weight'
            ]:
                f.write(f"'{key}' 未 merge（非目标权重）\n")

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, help='Path to the input model')
    parser.add_argument('--output_path', type=str, help='Path to the output model')
    parser.add_argument('--checkpoint_path', type=str, help='Path to the checkpoint directory')
    parser.add_argument('--log_file', type=str, default='merge_log.txt', help='Path to the log file')
    args = parser.parse_args()

    # 加载模型
    print('Loading model...')
    model = InternVLChatModel.from_pretrained(
        args.input_path, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16).eval()

    # 加载 tokenizer
    print('Loading tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(args.input_path, trust_remote_code=True)

    # 加载保存的权重
    print('Loading saved weights...')
    saved_weights = {}
    with safe_open(f'{args.checkpoint_path}/model-00001-of-00002.safetensors', framework="pt") as f:
        saved_weights.update({k: f.get_tensor(k) for k in f.keys()})
    with safe_open(f'{args.checkpoint_path}/model-00002-of-00002.safetensors', framework="pt") as f:
        saved_weights.update({k: f.get_tensor(k) for k in f.keys()})

    # 合并权重并记录日志
    print('Merging weights...')
    merge_weights_and_log(model, saved_weights, args.log_file)

    # 处理 LoRA 部分
    if model.config.use_backbone_lora:
        model.vision_model.merge_and_unload()
        model.vision_model = model.vision_model.model
        model.config.use_backbone_lora = 0
    if model.config.use_llm_lora:
        model.language_model.merge_and_unload()
        model.language_model = model.language_model.model
        model.config.use_llm_lora = 0

    # 保存模型
    print('Saving model...')
    model.save_pretrained(args.output_path)

    # 保存 tokenizer
    print('Saving tokenizer...')
    tokenizer.save_pretrained(args.output_path)

    print('Done!')

if __name__ == '__main__':
    main()