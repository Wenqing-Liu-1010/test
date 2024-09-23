import torch
from angle_emb import AnglE, Prompts
from angle_emb.utils import cosine_similarity

# 加载模型
angle = AnglE.from_pretrained('NousResearch/Llama-2-7b-hf',
                              pretrained_lora_path='SeanLee97/angle-llama-7b-nli-v2',
                              pooling_strategy='last',
                              is_llm=True,
                              torch_dtype=torch.float16).cuda()

# 打印所有预定义提示
print('All predefined prompts:', Prompts.list_prompts())

# 编码文本，不提供 prompt
doc_vecs = angle.encode([
    {'text': 'The weather is great!'},
    {'text': 'The weather is very good!'},
    {'text': 'I am going to bed'}
], prompt=None)  # 设置为 None

# 计算余弦相似度
for i, dv1 in enumerate(doc_vecs):
    for dv2 in doc_vecs[i+1:]:
        print(cosine_similarity(dv1, dv2))
