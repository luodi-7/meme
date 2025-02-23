from transformers import AutoModel

# 加载保存的模型
model = AutoModel.from_pretrained('/mnt/afs/xueyingyi/loc_version1/checkpoint-13200', trust_remote_code=True)
breakpoint()
# 检查 new_embedding 和 new_linear 的参数
print(model.language_model.model.model.embed_tokens)
print(model.language_model.lm_head)
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('/mnt/afs/xueyingyi/loc_version1/checkpoint-13200', add_eos_token=False, trust_remote_code=True, use_fast=False)