import tensorflow_text
import tensorflow_hub as hub
import tensorflow as tf
from angle_emb import AnglE, Prompts
from angle_emb.utils import cosine_similarity
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
import torch

class TextSimilarity:
    class AOEModel:
        def __init__(self, model_name='WhereIsAI/UAE-Large-V1', pooling_strategy='cls'):
            self.model = AnglE.from_pretrained(model_name, pooling_strategy=pooling_strategy).cuda()

        def encode_texts(self, texts):
            return self.model.encode(texts)

    class SimCSEModel:
        def __init__(self, model_name='princeton-nlp/sup-simcse-bert-base-uncased'):
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)

        def encode_texts(self, texts):
            inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
            with torch.no_grad():
                embeddings = self.model(**inputs, return_dict=True).pooler_output
            return embeddings

    class USEModel:
        def __init__(self):
            # 加载 Universal Sentence Encoder 多语言模型
            self.embedder = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3")

        def encode_texts(self, texts):
            # 对句子进行编码
            embeddings = self.embedder(texts)
            return embeddings

    class SBERTModel:
        def __init__(self, model_name='all-MiniLM-L6-v2'):
            # 加载 SBERT 模型
            self.model = SentenceTransformer(model_name)

        def encode_texts(self, texts):
            return self.model.encode(texts)

    class LLMModel:
        def __init__(self):
            # 加载 LLM (AnglE with LLaMA)
            self.model = AnglE.from_pretrained(
              'NousResearch/Llama-2-7b-hf',
              pretrained_lora_path='SeanLee97/angle-llama-7b-nli-v2',
              pooling_strategy='last',
              is_llm=True,
              torch_dtype=torch.float16,
              offload_dir='path/to/offload_dir'  # 指定 offload 目录
          ).cuda()


        def encode_texts(self, texts):
            prompts = [Prompts.A] * len(texts)
            doc_vecs = self.model.encode([{'text': text} for text in texts], prompt=prompts)
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
        else:
            raise ValueError("Invalid model_class. Choose 'aoe', 'simcse', 'use', 'sbert', or 'llm'.")

    def calculate_cosine_similarity(self, vec1, vec2):
        return cosine_similarity(vec1, vec2)

    def calculate_opposition_degree(self, similarity):
        return 1 - similarity


    def __call__(self, base_text):
        texts = [
            base_text + " The product's unique features attract a quite large customer base.",
            base_text + ' Competitors quickly release similar products, reducing the company’s advantage.',
            base_text + ' People frequently share their happy usage experience on social media.'
        ]

        doc_vecs = self.model.encode_texts(texts)

        # Calculate cosine similarities
        similarity_neutral_supporter = self.calculate_cosine_similarity(doc_vecs[2], doc_vecs[0])
        similarity_neutral_defeater = self.calculate_cosine_similarity(doc_vecs[2], doc_vecs[1])
        similarity_supporter_defeater = self.calculate_cosine_similarity(doc_vecs[0], doc_vecs[1])

        opposition_neutral_supporter = self.calculate_opposition_degree(similarity_neutral_supporter)
        opposition_neutral_defeater = self.calculate_opposition_degree(similarity_neutral_defeater)
        opposition_supporter_defeater = self.calculate_opposition_degree(similarity_supporter_defeater)

        # Print the similarities and opposition degrees
        print(f"Cosine similarity between Neutral and Supporter: {similarity_neutral_supporter:.4f}, Opposition Degree: {opposition_neutral_supporter:.4f}")
        print(f"Cosine similarity between Neutral and Defeater: {similarity_neutral_defeater:.4f}, Opposition Degree: {opposition_neutral_defeater:.4f}")
        print(f"Cosine similarity between Supporter and Defeater: {similarity_supporter_defeater:.4f}, Opposition Degree: {opposition_supporter_defeater:.4f}")


def main():
    base_text = 'A company launches a revolutionary product. The company gains a significant market share.'

    similarity_calculator_aoe = TextSimilarity(model_class='aoe', model_name='WhereIsAI/UAE-Large-V1')
    similarity_calculator_aoe(base_text)

    similarity_calculator_simcse_base = TextSimilarity(model_class='simcse', model_name='princeton-nlp/sup-simcse-bert-base-uncased')
    similarity_calculator_simcse_base(base_text)

    similarity_calculator_simcse_large = TextSimilarity(model_class='simcse', model_name='princeton-nlp/sup-simcse-bert-large-uncased')
    similarity_calculator_simcse_large(base_text)

    similarity_calculator_use = TextSimilarity(model_class='use')
    similarity_calculator_use(base_text)

    similarity_calculator_sbert = TextSimilarity(model_class='sbert', model_name='all-MiniLM-L6-v2')
    similarity_calculator_sbert(base_text)

    # similarity_calculator_llm = TextSimilarity(model_class='llm')
    # similarity_calculator_llm(base_text)

if __name__ == "__main__":
    main()
