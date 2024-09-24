import os
import numpy as np
import torch
import torch.nn as nn
import nltk

# 下载 NLTK 的 punkt tokenizer
nltk.download('punkt')

class InferSent(nn.Module):
    def __init__(self, config):
        super(InferSent, self).__init__()
        self.bsize = config['bsize']
        self.word_emb_dim = config['word_emb_dim']
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.pool_type = config['pool_type']
        self.dpout_model = config['dpout_model']
        self.version = config.get('version', 1)

        self.enc_lstm = nn.LSTM(self.word_emb_dim, self.enc_lstm_dim, 1,
                                bidirectional=True, dropout=self.dpout_model)

        # Initialize additional variables for version
        self.bos = '<s>'
        self.eos = '</s>'

    def forward(self, sent_tuple):
        sent, sent_len = sent_tuple
        # Implement the forward logic
        return sent  # Placeholder

    def set_w2v_path(self, w2v_path):
        self.w2v_path = w2v_path

    def build_vocab(self, sentences, tokenize=True):
        # Build vocabulary logic
        pass

    def encode(self, sentences, bsize=64):
        # Encode logic
        return np.random.rand(len(sentences), self.enc_lstm_dim * 2)  # Placeholder

# 设定模型参数
V = 2
MODEL_PATH = 'encoder/infersent%s.pkl' % V
params_model = {
    'bsize': 64,
    'word_emb_dim': 300,  # 嵌入维度
    'enc_lstm_dim': 2048,  # LSTM 隐藏层维度
    'pool_type': 'max',    # 池化类型
    'dpout_model': 0.0,     # Dropout
    'version': V
}

# 创建 InferSent 实例
infersent = InferSent(params_model)

# 加载模型权重
infersent.load_state_dict(torch.load(MODEL_PATH))

# 设置词向量路径并构建词汇表
W2V_PATH = 'fastText/crawl-300d-2M.vec'
infersent.set_w2v_path(W2V_PATH)

# 示例句子
sentences = ["This is a sentence.", "This is another sentence."]

# 构建词汇表
infersent.build_vocab(sentences, tokenize=True)

# 计算句子嵌入
embeddings = infersent.encode(sentences)

# 输出嵌入结果
print("Embeddings:")
print(embeddings)
