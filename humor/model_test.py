from transformers import RobertaTokenizerFast
from transformers import RobertaForSequenceClassification
from transformers import TextClassificationPipeline

model = RobertaForSequenceClassification.from_pretrained("Humor-Research/humor-detection-comb-23")
tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base", max_length=512, truncation=True)
pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, max_length=512, truncation=True)
from transformers import AutoConfig

config = AutoConfig.from_pretrained("Humor-Research/humor-detection-comb-23")
print(config.id2label)
print(pipe(["That joke is so funny"]))  # 幽默文本
print(pipe(["The weather is nice today"]))  # 非幽默文本
print(pipe(["Why don't scientists trust atoms? Because they make up everything!"])) #幽默
print(pipe(["I'm reading a book on anti-gravity. It's impossible to put down!"])) #双关
print(pipe(["I told my computer I needed a break, and now it won't stop sending me Kit-Kats."])) #反转
print(pipe(["I'm so good at sleeping, I can do it with my eyes closed."])) #夸张