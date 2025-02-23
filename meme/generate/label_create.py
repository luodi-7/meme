import pandas as pd

# 读取两个CSV文件
label_df = pd.read_csv('/mnt/afs/xueyingyi/meme/generate/label_E.csv',encoding='iso-8859-1')
text_df = pd.read_csv('/mnt/afs/xueyingyi/meme/generate/E_text.csv',encoding='iso-8859-1')

# 根据file_name列合并两个DataFrame
merged_df = pd.merge(label_df, text_df, on='file_name', how='left')

# 保存合并后的DataFrame到新的CSV文件
merged_df.to_csv('/mnt/afs/xueyingyi/meme/generate/generate_label.csv', index=False)

print("合并完成，结果已保存到 generate_label.csv")