import os
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
import tensorflow_hub as hub
from angle_emb import AnglE, Prompts
from huggingface_hub import login

login("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")


# 定义 Metrics 类
class Metrics:
    def __init__(self, op_sd, op_sn, op_dn):
        self.op_sd = （1-op_sd)/2
        self.op_sn = (1-op_sn)/2
        self.op_dn = (1-op_dn)/2

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

# 定义自定义 Dataset
class TextSimilarityDataset(Dataset):
    def __init__(self, file_path):
        self.data = pd.read_json(file_path, lines=True)
        self.supporters = (self.data['context_text'] + " " + self.data['supporter_text']).tolist()
        self.defeaters = (self.data['context_text'] + " " + self.data['defeater_text']).tolist()
        self.neutrals = (self.data['context_text'] + " " + self.data['neutral_text']).tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        supporter = self.supporters[idx]
        defeater = self.defeaters[idx]
        neutral = self.neutrals[idx]
        return supporter, defeater, neutral

# 定义 TextSimilarity 类
class TextSimilarity:
    class GTEModel:
        def __init__(self, model_name='Alibaba-NLP/gte-Qwen1.5-7B-instruct'):
            self.model = SentenceTransformer(model_name, trust_remote_code=True).cuda()
            self.model.max_seq_length = 8192  # Set maximum sequence length

        def encode_texts(self, texts):
            return self.model.encode(texts, convert_to_tensor=True).cuda()  # Ensure output is on CUDA
    
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
            self.model = AutoModel.from_pretrained(model_name).cuda()


        def encode_texts(self, texts, batch_size=64):
            embeddings = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to('cuda')
                with torch.no_grad():
                    outputs = self.model(**inputs, return_dict=True)
                    embeddings.append(outputs.pooler_output.cpu())
            return torch.cat(embeddings)

    class SBERTModel:
        def __init__(self, model_name='all-MiniLM-L6-v2'):
            self.model = SentenceTransformer(model_name).to('cuda')


        def encode_texts(self, texts, batch_size=64):
            embeddings = self.model.encode(texts, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=False)
            return embeddings.cpu()

    class LLMModel_7B:
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
            doc_vecs = self.model.encode([{'text': text} for text in texts], prompt=Prompts.A)
            return torch.tensor(doc_vecs).to('cuda')

    class LLMModel_13B:
        def __init__(self):
            self.model = AnglE.from_pretrained(
                'NousResearch/Llama-2-13b-hf',
                pretrained_lora_path='SeanLee97/angle-llama-13b-nli',
                pooling_strategy='last',
                is_llm=True,
                torch_dtype=torch.float16,
                offload_dir='path/to/offload_dir'
            ).cuda()


        def encode_texts(self, texts):
            doc_vecs = self.model.encode([{'text': text} for text in texts], prompt=Prompts.A)
            return torch.tensor(doc_vecs).to('cuda')

    
    class USEModel:
        def __init__(self):
            import tensorflow as tf
            import tensorflow_text  # 确保导入顺序正确
            self.embedder = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3")
            # TensorFlow Hub models run on CPU; to leverage GPU, consider using a different model

        def encode_texts(self, texts, batch_size=128):
            embeddings = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                embeddings_tf = self.embedder(batch)
                embeddings.append(torch.tensor(embeddings_tf.numpy()))
            return torch.cat(embeddings).to('cuda') if torch.cuda.is_available() else torch.cat(embeddings)

    class CoSENTModel:
        def __init__(self, model_name='shibing624/text2vec-base-multilingual'):
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)
            self.model = AutoModel.from_pretrained(model_name).cuda()

        def encode_texts(self, texts, batch_size=64):
            embeddings = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to('cuda')
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    doc_vecs = outputs.last_hidden_state.mean(dim=1).cpu()
                embeddings.append(doc_vecs)
            return torch.cat(embeddings)

    def __init__(self, model_class='gte', model_name=None):
        if model_class == 'gte':
            self.model = self.GTEModel()
        elif model_class == 'aoe':
            self.model = self.AOEModel(model_name=model_name)
        elif model_class == 'simcse':
            self.model = self.SimCSEModel(model_name=model_name)
        elif model_class == 'sbert':
            self.model = self.SBERTModel(model_name=model_name)
        elif model_class == 'llm_7B':
            self.model = self.LLMModel_7B()
        elif model_class == 'llm_13B':
            self.model = self.LLMModel_13B()
        elif model_class == 'use':
            self.model = self.USEModel()
        elif model_class == 'cosent':
            self.model = self.CoSENTModel(model_name=model_name)
        else:
            raise ValueError("Invalid model_class. Choose 'gte', 'aoe', 'simcse', 'sbert', 'llm_7B', 'llm_13B','use', or 'cosent'.")


    def calculate_cosine_similarity(self, vec1, vec2):
        if isinstance(vec1, np.ndarray):
            vec1 = torch.tensor(vec1).to('cuda')
        if isinstance(vec2, np.ndarray):
            vec2 = torch.tensor(vec2).to('cuda')

        return torch.cosine_similarity(vec1, vec2, dim=1)

    def __call__(self, dataloader):
        results = []
        for batch in dataloader:
            supporters, defeaters, neutrals = batch
            texts = supporters + defeaters + neutrals
            doc_vecs = self.model.encode_texts(texts)

            n = len(supporters)
            supporters_vecs = doc_vecs[:n]
            defeaters_vecs = doc_vecs[n:2*n]
            neutrals_vecs = doc_vecs[2*n:3*n]

            similarity_neutral_supporter = self.calculate_cosine_similarity(neutrals_vecs, supporters_vecs)
            similarity_neutral_defeater = self.calculate_cosine_similarity(neutrals_vecs, defeaters_vecs)
            similarity_supporter_defeater = self.calculate_cosine_similarity(supporters_vecs, defeaters_vecs)

            results.extend([
                {
                    "neutral_supporter_similarity": sim_neut_sup.item(),
                    "neutral_defeater_similarity": sim_neut_def.item(),
                    "supporter_defeater_similarity": sim_sup_def.item(),
                }
                for sim_neut_sup, sim_neut_def, sim_sup_def in zip(
                    similarity_neutral_supporter,
                    similarity_neutral_defeater,
                    similarity_supporter_defeater
                )
            ])

        results_df = pd.DataFrame(results)
        return results_df

# 定义主函数
def main():
    # 加载数据集
    file_path = '/mnt/lia/scratch/wenqliu/evaluation/delta_causal/test_processed_filtered.jsonl'
    if not os.path.exists(file_path):
        print(f"数据文件未找到，请确保路径正确：{file_path}")
        return
    dataset = TextSimilarityDataset(file_path)

    # 定义 DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=16,       # 根据您的 GPU 内存调整批次大小
        shuffle=False,        # 如果需要打乱数据，可以设置为 True
        num_workers=4,        # 根据您的 CPU 核心数调整
        pin_memory=True if torch.cuda.is_available() else False  # 如果使用 GPU，加快数据传输速度
    )

    # 定义模型及其对应的名称
    models = [
        #('gte', None),
        ('aoe', 'WhereIsAI/UAE-Large-V1'),
        ('simcse', 'princeton-nlp/sup-simcse-bert-base-uncased'),
        ('sbert', 'all-MiniLM-L6-v2'),
        ('cosent', 'shibing624/text2vec-base-multilingual'),
        ('llm_7B', None),  # LLM does not require a model name
        ('llm_13B', None),  # LLM does not require a model name
        ('use', None),
    ]

    # 遍历每个模型，计算相似度
    for model_class, model_name in models:
        print(f"Using {model_class} model...")
        try:
            similarity_calculator = TextSimilarity(model_class=model_class, model_name=model_name)
            results_df = similarity_calculator(dataloader)

            # 将结果添加到原始数据中
            dataset.data[f'neutral_supporter_similarity_{model_class}'] = results_df['neutral_supporter_similarity'].values
            dataset.data[f'neutral_defeater_similarity_{model_class}'] = results_df['neutral_defeater_similarity'].values
            dataset.data[f'supporter_defeater_similarity_{model_class}'] = results_df['supporter_defeater_similarity'].values

            # 保存更新的数据
            output_dir = '/mnt/lia/scratch/wenqliu/evaluation/delta_causal/existing_models/'
            os.makedirs(output_dir, exist_ok=True)
            output_file_path = os.path.join(output_dir, f'{model_class}_results_filtered.jsonl')
            dataset.data.to_json(output_file_path, orient='records', lines=True)
            print(f"Results for {model_class} model have been saved to: {output_file_path}")

            # 计算指标：DCF、DOW 等
            metrics_calculator = Metrics(
                op_sd=dataset.data[f'neutral_supporter_similarity_{model_class}'].values,
                op_sn=dataset.data[f'neutral_defeater_similarity_{model_class}'].values,
                op_dn=dataset.data[f'supporter_defeater_similarity_{model_class}'].values
            )

            dcf_value = metrics_calculator.DCF()
            dcf_positive_value = metrics_calculator.DCF_positive()
            dcf_negative_value = metrics_calculator.DCF_negative()
            dow_value = metrics_calculator.DOW()
            or_value = metrics_calculator.OR()

            # 输出计算结果
            print(f"{model_class} DCF: {dcf_value:.4f}")
            print(f"{model_class} DCF_positive: {dcf_positive_value:.4f}")
            print(f"{model_class} DCF_negative: {dcf_negative_value:.4f}")
            print(f"{model_class} DOW: {dow_value:.4f}")
            print(f"{model_class} OR: {or_value:.4f}")
            print("\n" + "="*40 + "\n")  # 分隔每个模型的结果

        except Exception as e:
            print(f"在使用模型 {model_class} 时发生错误: {e}")
            print("\n" + "="*40 + "\n")  # 分隔每个模型的结果

if __name__ == "__main__":
    main()
