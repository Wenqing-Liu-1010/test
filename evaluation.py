import torch
from angle_emb import AnglE
from angle_emb.utils import cosine_similarity
import pandas as pd
import numpy as np
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
import tensorflow_hub as hub

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
            self.model = AutoModel.from_pretrained(model_name).cuda()

        def encode_texts(self, texts):
            inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to('cuda')
            with torch.no_grad():
                embeddings = self.model(**inputs, return_dict=True).pooler_output
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
            prompts = [Prompts.A] * len(texts)  # Ensure Prompts class is defined
            doc_vecs = self.model.encode([{'text': text} for text in texts], prompt=prompts)
            return doc_vecs

    class USEModel:
        def __init__(self):
            # Load the Universal Sentence Encoder multilingual model
            self.embedder = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3")

        def encode_texts(self, texts):
            # Encode sentences
            embeddings = self.embedder(texts)
            return embeddings

    def __init__(self, model_class='aoe', model_name=None):
        if model_class == 'aoe':
            self.model = self.AOEModel(model_name=model_name)
        elif model_class == 'simcse':
            self.model = self.SimCSEModel(model_name=model_name)
        elif model_class == 'sbert':
            self.model = self.SBERTModel(model_name=model_name)
        elif model_class == 'llm':
            self.model = self.LLMModel()
        elif model_class == 'use':
            self.model = self.USEModel()
        else:
            raise ValueError("Invalid model_class. Choose 'aoe', 'simcse', 'sbert', 'llm', or 'use'.")

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

                # Calculate similarities
                similarity_neutral_supporter = self.calculate_cosine_similarity(doc_vecs[2], doc_vecs[0])
                similarity_neutral_defeater = self.calculate_cosine_similarity(doc_vecs[2], doc_vecs[1])
                similarity_supporter_defeater = self.calculate_cosine_similarity(doc_vecs[0], doc_vecs[1])

                results.append({
                    "neutral_supporter_similarity": similarity_neutral_supporter.item() if hasattr(similarity_neutral_supporter, 'item') else similarity_neutral_supporter,
                    "neutral_defeater_similarity": similarity_neutral_defeater.item() if hasattr(similarity_neutral_defeater, 'item') else similarity_neutral_defeater,
                    "supporter_defeater_similarity": similarity_supporter_defeater.item() if hasattr(similarity_supporter_defeater, 'item') else similarity_supporter_defeater,
                })

        results_df = pd.DataFrame(results)

        for column in results_df.columns:
            data[column] = results_df[column].values

        return data

def main():
    # Load the dataset
    file_path = '/mnt/lia/scratch/wenqliu/evaluation/test_processed_filtered.jsonl'
    data = pd.read_json(file_path, lines=True)

    # Define models and their corresponding names
    models = [
        # ('aoe', 'WhereIsAI/UAE-Large-V1'),
        ('simcse', 'princeton-nlp/sup-simcse-bert-base-uncased'),
        ('sbert', 'all-MiniLM-L6-v2'),
        ('llm', None),  # LLM does not require a model name
        ('use', None)   # USE does not require a model name
    ]

    # Iterate over each model to calculate similarities
    for model_class, model_name in models:
        print(f"Using {model_class} model...")
        similarity_calculator = TextSimilarity(model_class=model_class, model_name=model_name)
        updated_data = similarity_calculator(data, batch_size=2048)

        # Save the updated data
        output_file_path = f'/mnt/lia/scratch/wenqliu/evaluation/existing_models/{model_class}_results.jsonl'
        updated_data.to_json(output_file_path, orient='records', lines=True)
        print(f"Results for {model_class} model have been saved to: {output_file_path}")

        # Calculate metrics: DCF, DOW, etc.
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

        # Print the calculated metrics
        print(f"{model_class} DCF: {dcf_value:.4f}")
        print(f"{model_class} DCF_positive: {dcf_positive_value:.4f}")
        print(f"{model_class} DCF_negative: {dcf_negative_value:.4f}")
        print(f"{model_class} DOW: {dow_value:.4f}")
        print(f"{model_class} OR: {or_value:.4f}")
        print("\n" + "="*40 + "\n")  # Separator for each model's results

if __name__ == "__main__":
    main()
