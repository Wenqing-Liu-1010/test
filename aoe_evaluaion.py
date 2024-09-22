import torch
from angle_emb import AnglE
from angle_emb.utils import cosine_similarity

import pandas as pd

file_path = '/mnt/lia/scratch/yifeng/dichotomous-score/data/defeasible_snli/test_processed.jsonl'
data = pd.read_json(file_path, lines=True)



class TextSimilarity:
    class AOEModel:
        def __init__(self, model_name='WhereIsAI/UAE-Large-V1', pooling_strategy='cls'):
            self.model = AnglE.from_pretrained(model_name, pooling_strategy=pooling_strategy).cuda()

        def encode_texts(self, texts):
            return self.model.encode(texts)

    def __init__(self, model_class='aoe', model_name=None):
        if model_class == 'aoe':
            self.model = self.AOEModel(model_name=model_name)
        else:
            raise ValueError("Invalid model_class. Choose 'aoe'.")

    def calculate_cosine_similarity(self, vec1, vec2):
        return cosine_similarity(vec1, vec2)

    def __call__(self, data):
        results = []
        for index in range(min(100, len(data))):  # 取前100条数据
            row = data.iloc[index]
            context = row['context_text']
            supporter = context + " " + row['supporter_text']  # 连接上下文和支持者
            defeater = context + " " + row['defeater_text']  # 连接上下文和反对者
            neutral = context + " " + row['neutral_text']      # 连接上下文和中立文本

            texts = [supporter, defeater, neutral]
            doc_vecs = self.model.encode_texts(texts)

            # 计算相似度
            similarity_neutral_supporter = self.calculate_cosine_similarity(doc_vecs[2], doc_vecs[0])  # neutral vs supporter
            similarity_neutral_defeater = self.calculate_cosine_similarity(doc_vecs[2], doc_vecs[1])  # neutral vs defeater
            similarity_supporter_defeater = self.calculate_cosine_similarity(doc_vecs[0], doc_vecs[1])  # supporter vs defeater

            results.append({
                "neutral_supporter_similarity": similarity_neutral_supporter,
                "neutral_defeater_similarity": similarity_neutral_defeater,
                "supporter_defeater_similarity": similarity_supporter_defeater,
            })

        return results
print(results)
