import pandas as pd
import numpy as np
import torch
import tensorflow_text
import tensorflow_hub as hub
import tensorflow as tf
from angle_emb import AnglE, Prompts
from angle_emb.utils import cosine_similarity
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer

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
            self.model = AutoModel.from_pretrained(model_name)

        def encode_texts(self, texts):
            inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
            inputs = {k: v.cuda() for k, v in inputs.items()}  # 确保输入张量在 GPU 上
            with torch.no_grad():
                embeddings = self.model(**inputs, return_dict=True).pooler_output
            return embeddings

    class USEModel:
        def __init__(self):
            self.embedder = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3")

        def encode_texts(self, texts):
            embeddings = self.embedder(texts)
            return embeddings

    class SBERTModel:
        def __init__(self, model_name='all-MiniLM-L6-v2'):
            self.model = SentenceTransformer(model_name)

        def encode_texts(self, texts):
            return self.model.encode(texts)

    class LLMModel:
        def __init__(self):
            self.model = AnglE.from_pretrained(
                'NousResearch/Llama-2-7b-hf',
                pretrained_lora_path='SeanLee97/angle-llama-7b-nli-v2',
                pooling_strategy='last',
                is_llm=True,
                torch_dtype=torch.float16,
                offload_dir='path/to/offload_dir'
            ).cuda()

        def encode_texts(self, texts):
            prompts = [Prompts.A] * len(texts)
            doc_vecs = self.model.encode([{'text': text} for text in texts], prompt=prompts)
            return doc_vecs

    class CoSENTModel:
        def __init__(self, model_name='shibing624/text2vec-base-multilingual'):
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)

        def encode_texts(self, texts):
            inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
            inputs = {k: v.cuda() for k, v in inputs.items()}  # 确保输入张量在 GPU 上
            with torch.no_grad():
                doc_vecs = self.model(**inputs).last_hidden_state.mean(dim=1)
            return doc_vecs

    def __init__(self, model_class='aoe', model_name=None):
        if model_class == 'aoe':
            self.model = self.AOEModel(model_name=model_name)
        elif model_class == 'simcse':
            self.model = self.SimCSEModel(model_name=model_name)
        elif model_class == 'use':
            self.model = self.USEModel()
        elif model_class == 'sbert':
            self.model = self.SBERTModel(model_name=model_name)
        elif model_class == 'llm':
            self.model = self.LLMModel()
        elif model_class == 'cosent':
            self.model = self.CoSENTModel(model_name=model_name)
        else:
            raise ValueError("Invalid model_class. Choose 'aoe', 'simcse', 'use', 'sbert', 'llm', or 'cosent'.")

    def calculate_cosine_similarity(self, vec1, vec2):
        return cosine_similarity(vec1, vec2)

    def calculate_opposition_degree(self, similarity):
        return 1 - similarity

    def __call__(self, data):
        results = []
        for index in range(min(100, len(data))):  # 取前100条数据
            row = data.iloc[index]
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
                "neutral_supporter_similarity": similarity_neutral_supporter,
                "neutral_defeater_similarity": similarity_neutral_defeater,
                "supporter_defeater_similarity": similarity_supporter_defeater,
            })

        return results

def main():
    # 读取数据
    file_path = '/mnt/lia/scratch/yifeng/dichotomous-score/data/defeasible_snli/test_processed.jsonl'
    data = pd.read_json(file_path, lines=True)

    # 创建相似度计算器并计算结果
    similarity_calculator = TextSimilarity(model_class='simcse', model_name='princeton-nlp/sup-simcse-bert-base-uncased')
    similarity_results = similarity_calculator(data)

    # 转换为 DataFrame
    results_df = pd.DataFrame(similarity_results)

    # 计算每列的平均值
    averages = results_df.mean()
    print("每一列的平均值:")
    print(averages)

    # 计算 DCF、DOW 等指标
    metrics_calculator = Metrics(
        op_sd=results_df['neutral_supporter_similarity'].values,
        op_sn=results_df['neutral_defeater_similarity'].values,
        op_dn=results_df['supporter_defeater_similarity'].values
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
