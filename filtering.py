import pandas as pd
import os
import torch
from angle_emb import AnglE
from transformers import AutoModel, AutoTokenizer

# Dataset names
splits = ['train', 'dev', 'test']  # Modify based on actual dataset names

# Ensure the device is set to CUDA if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Loop through each dataset
for split in splits:
    # Read the data
    file_path = f'/mnt/lia/scratch/yifeng/dichotomous-score/data/defeasible_snli/{split}_processed.jsonl'
    data = pd.read_json(file_path, lines=True)

    # Create a boolean mask to check if the first five characters of neutral_id match supporter_id or defeater_id
    mask = (data['neutral_id'].str[:5] == data['supporter_id'].str[:5]) | \
           (data['neutral_id'].str[:5] == data['defeater_id'].str[:5])

    # Filter the data to keep only the rows that satisfy the condition
    filtered_data = data[mask]

    # Save the filtered data to a new file in a directory you have permission for
    output_file_path = f'/mnt/lia/scratch/wenqliu/evaluation/{split}_processed_filtered.jsonl'
    filtered_data.to_json(output_file_path, orient='records', lines=True)
    print(f"{split} dataset has been processed, results saved to: {output_file_path}")

    # If using a model, make sure to load it to the correct device
    # model = YourModelClass().to(device)  # Example of loading a model to GPU
