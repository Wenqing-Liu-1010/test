import torch
from angle_emb import AnglE
from angle_emb.utils import cosine_similarity
import pandas as pd
import numpy as np
from transformers import AutoModel, AutoTokenizer

class Metrics:
    def __init__(self, op_sd, op_sn, op_dn):
        self.op_sd = op_sd
        self.op_sn = op_sn
        self.op_dn = op_dn

    def DCF(self):
        N = len(self.op_sd)
        count = sum((self.op_sd[i] > self.op_sn[i]) and (self.op_sd[i] > self.op_dn[i]) for i in range(N))
        return count / N

    def DCF_positive(self):
        N = len(self.op_sd)
        count = sum((self.op_sd[i] > self.op_sn[i]) for i in range(N))
        return count / N

    def DCF_negative(self):
        N = len(self.op_sd)
        count = sum((self.op_sd[i] > self.op_dn[i]) for i in range(N))
        return count / N

    def DOW(self):
        DOW_value = self.op_sd - np.maximum(self.op_sn, self.op_dn)
        return DOW_value.mean()

    def OR(self):
        OR_value = self.op_sd / np.maximum(self.op_sn, self.op_dn)
        return OR_value.mean()

class TextSimilarity:
    class AOEModel:
        def __init__(self, model_name='WhereIsAI/UAE-Large-V1', pooling_strategy='cls'):
            if model_name is None:
                raise ValueError("model_name cannot be None")
            self.model = AnglE.from_pretrained(model_name, pooling_strategy=pooling_strategy).cuda()

        def encode_texts(self, texts):
            return self.model.encode(texts)

    class SimCSEModel:
        def __init__(self, model_name='princeton-nlp/sup-simcse-bert-base-uncased'):
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).cuda()  # 移到 GPU

        def encode_texts(self, texts):
            inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to('cuda')  # 移到 GPU
            with torch.no_grad():
                embeddings = self.model(**inputs, return_dict=True).pooler_output
            return embeddings

    def __init__(self, model_class='aoe', model_name=None):
        if model_class == 'aoe':
            self.model = self.AOEModel(model_name=model_name)
        elif model_class == 'simcse':
            self.model = self.SimCSEModel(model_name=model_name)

    def calculate_cosine_similarity(self, vec1, vec2):
        return cosine_similarity(vec1, vec2)

    def __call__(self, data, batch_size=32):
        results = []
        for start in range(0, len(data), batch_size):
            end = min(start + batch_size, len(data))
            batch = data.iloc[start:end]

            for index in range(len(batch)):
                row = batch.iloc[index]
                context = row['context_text']
                supporter = context + " " + row['supporter_text']
                defeater = context + " " + row['defeater_text']
                neutral = context + " " + row['neutral_text']

                texts = [supporter, defeater, neutral]
                doc_vecs = self.model.encode_texts(texts)

                # 计算相似度
                similarity_neutral_supporter = self.calculate_cosine_similarity(doc_vecs[2], doc_vecs[0])  # neutral vs supporter
                similarity_neutral_defeater = self.calculate_cosine_similarity(doc_vecs[2], doc_vecs[1])  # neutral vs defeater
                similarity_supporter_defeater = self.calculate_cosine_similarity(doc_vecs[0], doc_vecs[1])  # supporter vs defeater

                results.append({
                    "neutral_supporter_similarity": similarity_neutral_supporter.item() if hasattr(similarity_neutral_supporter, 'item') else similarity_neutral_supporter,
                    "neutral_defeater_similarity": similarity_neutral_defeater.item() if hasattr(similarity_neutral_defeater, 'item') else similarity_neutral_defeater,
                    "supporter_defeater_similarity": similarity_supporter_defeater.item() if hasattr(similarity_supporter_defeater, 'item') else similarity_supporter_defeater,
                })

        # 将计算结果转换为 DataFrame
        results_df = pd.DataFrame(results)

        # 将计算得到的相似度值放回原数据集中
        for column in results_df.columns:
            data[column] = results_df[column].values

        return data

def main():
    # 读取数据
    file_path = '/mnt/lia/scratch/yifeng/dichotomous-score/data/defeasible_snli/test_processed.jsonl'  # 确保路径正确
    data = pd.read_json(file_path, lines=True)

    # 创建相似度计算器并计算结果
    similarity_calculator = TextSimilarity(model_class='simcse', model_name='princeton-nlp/sup-simcse-bert-base-uncased')
    updated_data = similarity_calculator(data, batch_size=2048)

    # 保存更新后的数据
    output_file_path = '/mnt/lia/scratch/wenqliu/evaluation/simcse_bert.jsonl'  # 设定保存路径
    updated_data.to_json(output_file_path, orient='records', lines=True)
    print(f"更新后的数据已保存到: {output_file_path}")

    # 计算 DCF、DOW 等指标
    metrics_calculator = Metrics(
        op_sd=updated_data['neutral_supporter_similarity'].values,
        op_sn=updated_data['neutral_defeater_similarity'].values,
        op_dn=updated_data['supporter_defeater_similarity'].values
    )

    dcf_value = metrics_calculator.DCF()
    dcf_positive_value = metrics_calculator.DCF_positive()
    dcf_negative_value = metrics_calculator.DCF_negative()
    dow_value = metrics_calculator.DOW()
    or_value = metrics_calculator.OR()

    # 输出指标结果
    print(f"DCF: {dcf_value:.4f}")
    print(f"DCF_positive: {dcf_positive_value:.4f}")
    print(f"DCF_negative: {dcf_negative_value:.4f}")
    print(f"DOW: {dow_value:.4f}")
    print(f"OR: {or_value:.4f}")

if __name__ == "__main__":
    main()
