import pandas as pd
import os

# Dataset names
splits = ['train', 'dev', 'test']  # Modify based on actual dataset names

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

    # Save the filtered data to a new file
    output_file_path = f'/mnt/lia/scratch/yifeng/dichotomous-score/data/defeasible_snli/{split}_processed_filtered.jsonl'
    filtered_data.to_json(output_file_path, orient='records', lines=True)
    print(f"{split} dataset has been processed, results saved to: {output_file_path}")
