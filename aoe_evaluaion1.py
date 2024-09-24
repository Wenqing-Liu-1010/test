import os
import numpy as np
import pandas as pd
from huggingface_hub import login

# 登录 Hugging Face Hub
login("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")

# 定义要加载的模型列表
# models = [
#     'aoe', 'cosent', 'llm_13B', 'llm', 'sbert', 'use'
# ]
# 遍历每个模型
for model_class in models:
    file_path = f'/mnt/lia/scratch/wenqliu/evaluation/delta_causal/existing_models/{model_class}_results.jsonl'
    
    if not os.path.exists(file_path):
        print(f"数据文件未找到，请确保路径正确：{file_path}")
        continue

    data = pd.read_json(file_path, lines=True)

    # 获取相似度列
    op_sd = (1 - data[f'supporter_defeater_similarity_{model_class}'].values) / 2
    op_sn = (1 - data[f'neutral_supporter_similarity_{model_class}'].values) / 2
    op_dn = (1 - data[f'neutral_defeater_similarity_{model_class}'].values) / 2

    # 计算 DCF
    count_dcf = sum((op_sd[i] > op_sn[i]) and (op_sd[i] > op_dn[i]) for i in range(len(op_sd)))
    dcf_value = count_dcf / len(op_sd)

    # 计算 DCF_positive
    count_dcf_positive = sum(op_sd[i] > op_sn[i] for i in range(len(op_sd)))
    dcf_positive_value = count_dcf_positive / len(op_sd)

    # 计算 DCF_negative
    count_dcf_negative = sum(op_sd[i] > op_dn[i] for i in range(len(op_sd)))
    dcf_negative_value = count_dcf_negative / len(op_sd)


    # 输出结果
    print(f"Model: {model_class}")
    print(f"DCF: {dcf_value:.4f}")
    print(f"DCF_positive: {dcf_positive_value:.4f}")
    print(f"DCF_negative: {dcf_negative_value:.4f}")
    print("\n" + "="*40 + "\n")  # 分隔每个模型的结果
