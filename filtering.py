import pandas as pd
import os
import torch

# #####Senario B

splits = ['train', 'dev','test']  # Modify based on actual dataset names
#splits = ['test']  # Modify based on actual dataset names
# Ensure the device is set to CUDA if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Loop through each dataset
for split in splits:
    # Read the data
    file_path = f'/mnt/lia/scratch/yifeng/dichotomous-score/data/defeasible_snli/{split}_processed.jsonl'
    data = pd.read_json(file_path, lines=True)

    # Print the shape before filtering
    print(f"{split} dataset shape before filtering: {data.shape}")

    # Check if ". Hypothesis:" exists and add "." if needed
    mask_hypothesis = data['context_text'].str.contains(r'\. Hypothesis:')
    data.loc[~mask_hypothesis, 'context_text'] = data.loc[~mask_hypothesis, 'context_text'].str.replace('Hypothesis:', '. Hypothesis:')

    # Replace " . Hypothesis" with ". Hypothesis"
    data['context_text'] = data['context_text'].str.replace(' . Hypothesis', '. Hypothesis')

    # Extract neutral_id_prefix
    data['neutral_id_prefix'] = data['neutral_id'].str.extract(r'([^_]*_[^_]*)')[0]

    # Create a boolean mask to check if neutral_id_prefix matches supporter_id or defeater_id
    mask = (data['neutral_id_prefix'] == data['supporter_id']) | \
           (data['neutral_id_prefix'] == data['defeater_id'])

    # Filter the data to keep only the rows that satisfy the condition
    filtered_data = data[mask]

    # Print the shape after filtering
    print(f"{split} dataset shape after filtering: {filtered_data.shape}")

    # Save the filtered data to a new file in a directory you have permission for
    output_file_path = f'/mnt/lia/scratch/wenqliu/evaluation/{split}_processed_filtered.jsonl'
    filtered_data.to_json(output_file_path, orient='records', lines=True)
    print(f"{split} dataset has been processed, results saved to: {output_file_path}")



# #####Senario A
# import pandas as pd
# import torch

# # Dataset names
# splits = ['train','dev', 'test']  # Modify based on actual dataset names

# # Ensure the device is set to CUDA if available
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # Loop through each dataset
# for split in splits:
#     # Read the data
#     file_path = f'/mnt/lia/scratch/yifeng/dichotomous-score/data/perspectrum/{split}_processed.jsonl'
#     data = pd.read_json(file_path, lines=True)

#     # Print the shape before processing
#     print(f"{split} dataset shape before processing: {data.shape}")

#     # Function to ensure text ends with a period
#     def ensure_period(text):
#         return text if text.endswith('.') else text + '.'

#     # Apply the function to the relevant text columns
#     for column in ['context_text', 'supporter_text', 'defeater_text', 'neutral_text']:
#         data[column] = data[column].apply(ensure_period)

#     # Extract neutral_id_prefix
#     data['neutral_id_prefix'] = data['neutral_id'].str.extract(r'([^_]*_[^_]*)')[0]

#     # Create a boolean mask to check if neutral_id_prefix matches supporter_id or defeater_id
#     mask = (data['neutral_id_prefix'] == data['supporter_id']) | \
#            (data['neutral_id_prefix'] == data['defeater_id'])

#     # Filter the data to keep only the rows that satisfy the condition
#     filtered_data = data[mask]

#     # Print the shape after filtering
#     print(f"{split} dataset shape after filtering: {filtered_data.shape}")

#     # Save the filtered data to a new file in a directory you have permission for
#     output_file_path = f'/mnt/lia/scratch/wenqliu/evaluation/perspectrum/{split}_processed_filtered.jsonl'
#     filtered_data.to_json(output_file_path, orient='records', lines=True)
#     print(f"{split} dataset has been processed, results saved to: {output_file_path}")


# #####Senario C

# splits = ['train', 'dev','test']  # Modify based on actual dataset names
# #splits = ['test']  # Modify based on actual dataset names
# # Ensure the device is set to CUDA if available
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # Loop through each dataset
# for split in splits:
#     # Read the data
#     file_path = f'/mnt/lia/scratch/yifeng/dichotomous-score/data/delta_causal/{split}_processed.jsonl'
#     data = pd.read_json(file_path, lines=True)

#     # Print the shape before filtering
#     print(f"{split} dataset shape before filtering: {data.shape}")

#     # Check if ". Hypothesis:" exists and add "." if needed
#     mask_hypothesis = data['context_text'].str.contains(r'\. Effect:')
#     data.loc[~mask_hypothesis, 'context_text'] = data.loc[~mask_hypothesis, 'context_text'].str.replace('Effect:', '. Effect:')

#     # Replace " . Hypothesis" with ". Hypothesis"
#     data['context_text'] = data['context_text'].str.replace(' . Effect', '. Effect')

#     # Extract neutral_id_prefix
#     data['neutral_id_prefix'] = data['neutral_id'].str.extract(r'([^_]*_[^_]*)')[0]

#     # Create a boolean mask to check if neutral_id_prefix matches supporter_id or defeater_id
#     mask = (data['neutral_id_prefix'] == data['supporter_id']) | \
#            (data['neutral_id_prefix'] == data['defeater_id'])

#     # Filter the data to keep only the rows that satisfy the condition
#     filtered_data = data[mask]

#     # Print the shape after filtering
#     print(f"{split} dataset shape after filtering: {filtered_data.shape}")

#     # Save the filtered data to a new file in a directory you have permission for
#     output_file_path = f'/mnt/lia/scratch/wenqliu/evaluation/{split}_processed_filtered.jsonl'
#     filtered_data.to_json(output_file_path, orient='records', lines=True)
#     print(f"{split} dataset has been processed, results saved to: {output_file_path}")

