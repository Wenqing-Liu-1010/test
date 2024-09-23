import torch
from angle_emb import AnglE, Prompts
from angle_emb.utils import cosine_similarity

# 加载模型
angle = AnglE.from_pretrained(
    'NousResearch/Llama-2-7b-hf',
    pretrained_lora_path='SeanLee97/angle-llama-7b-nli-v2',
    pooling_strategy='last',
    is_llm=True,
    torch_dtype=torch.float16
).cuda()

# 打印所有预定义提示
print('All predefined prompts:', Prompts.list_prompts())

# 编码文本，传递 prompt
doc_vecs = angle.encode([
    {'text': 'The weather is great!'},
    {'text': 'The weather is very good!'},
    {'text': 'I am going to bed'}
], prompt=Prompts.A)  # 使用预定义的 prompt

# 计算余弦相似度
for i in range(len(doc_vecs)):
    for j in range(i + 1, len(doc_vecs)):
        similarity = cosine_similarity(doc_vecs[i], doc_vecs[j])
        print(f"Cosine similarity between doc {i} and doc {j}: {similarity.item():.4f}")
