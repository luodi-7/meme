import torch
from transformers import AutoTokenizer
from FlagEmbedding import FlagModel
from sentence_transformers import util
from transformers import pipeline  # 用于流畅性评估
from transformers import RobertaForSequenceClassification, RobertaTokenizerFast
class SimilarityModel():
    def __init__(self):
        model_path = '/mnt/afs/share/bge-base-zh-v1.5'
        self.model = FlagModel(model_path, 
                              query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                              use_fp16=True)

    def compare_similarity(self, output_str, label_str):
        embeddings_output = self.model.encode(output_str)
        embeddings_label = self.model.encode(label_str)
        cosine_scores = util.cos_sim(embeddings_output, embeddings_label).item()
        return cosine_scores

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

class FluencyModel():
    def __init__(self):
        # 加载 GPT-2 模型和分词器
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model.eval()  # 设置为评估模式

    def evaluate_fluency(self, text):
        # 使用 GPT-2 计算困惑度
        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss  # 交叉熵损失
            perplexity = torch.exp(loss).item()  # 困惑度 = exp(loss)
        
        # 将困惑度归一化到 0 到 1 的范围
        normalized_score = self.normalize_fluency_score(perplexity)
        return normalized_score

    def normalize_fluency_score(self, perplexity):
        # 假设困惑度的范围是 10 到 1000
        min_perplexity = 10
        max_perplexity = 1000
        
        # 将困惑度映射到 0 到 1 的范围
        normalized_score = 1 - (perplexity - min_perplexity) / (max_perplexity - min_perplexity)
        
        # 确保得分在 0 到 1 之间
        normalized_score = max(0, min(1, normalized_score))
        return normalized_score

class HumorModel():
    def __init__(self):
        # 手动加载模型和分词器
        self.model = RobertaForSequenceClassification.from_pretrained("Humor-Research/humor-detection-comb-23")
        self.tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base", max_length=512, truncation=True)

    def evaluate_humor(self, text):
        # 使用分词器对输入文本进行编码
        inputs = self.tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
        
        # 使用模型进行预测
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)  # 将 logits 转换为概率
        
        # 获取预测标签和置信度分数
        label_id = torch.argmax(probs, dim=-1).item()  # 预测的标签 ID (0 或 1)
        score = probs[0][label_id].item()  # 预测标签的置信度分数

        # 计算幽默得分
        if label_id == 1:  # LABEL_1 表示幽默
            humor_score = score
        else:  # LABEL_0 表示非幽默
            humor_score = 1 - score

        return humor_score
def compute_json_metric():
    # 初始化相似度模型、流畅性模型和幽默感模型
    simmodel = SimilarityModel()
    fluency_model = FluencyModel()
    humor_model = HumorModel()
    result = ["THEY HAVEN'T NOTICED......YET", "We review our ideas Thank you 2017"]
    label = ["THEY HAVEN'T NOTICED......YET", "We review our ideas Thank you 2017"]
    # 计算相似度、流畅性和幽默感
    similarity_scores = []
    fluency_scores = []
    humor_scores = []
    for r, l in zip(result, label):
        similarity_score = simmodel.compare_similarity(r, l)
        fluency_score = fluency_model.evaluate_fluency(r)
        humor_score = humor_model.evaluate_humor(r)
        similarity_scores.append(similarity_score)
        fluency_scores.append(fluency_score)
        humor_scores.append(humor_score)
        
        # 计算平均相似度、流畅性和幽默感
    avg_similarity = sum(similarity_scores) / len(similarity_scores)
    avg_fluency = sum(fluency_scores) / len(fluency_scores)
    avg_humor = sum(humor_scores) / len(humor_scores)
        
    # 返回一个字典，Trainer 会将其记录到日志中
    return {
        "avg_similarity": avg_similarity,
        "avg_fluency": avg_fluency,
        "avg_humor": avg_humor
    }

        

if __name__ == '__main__':
    print(compute_json_metric())