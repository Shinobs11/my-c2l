import json, os, pandas as pd
from src.utils.pickleUtils import pload, pdump
import ast
from src.classes.datasets  import CFClassifcationDataset
from transformers import BertTokenizerFast
import torch
def splitAugmentedSet(dataset_name:str):
  base_path = f"datasets/{dataset_name}"
  aug_base_path = f"{base_path}/cl"
  base_base_path = f"{base_path}/base"
  aug_set_path = f"{aug_base_path}/augmented_triplets.json"
  aug_set_file = open(aug_set_path, mode='r')
  aug_set = json.load(aug_set_file)

  aug_dict = {
    "anchor_text": [],
    "positive_text": [],
    "negative_text": [],
    "label": [],
    "triplet_sample_mask": []
  }
  for x in aug_set:
    for k, v in aug_dict.items():
      v.append(x[k])
  
  aug_df = pd.DataFrame.from_dict(aug_dict)
  
  train_set = aug_df



  train_set['label'] = [ast.literal_eval(str(x)) for x in train_set['label']]


    
  tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
  meta = torch.load(f"{base_path}/dataset_meta.pt")
  token_limit = meta['token_limit']
  train_anchor_encodings = tokenizer(train_set['anchor_text'].tolist(), padding='max_length', truncation=True, max_length=token_limit)
  train_positive_encodings = tokenizer(train_set['positive_text'].tolist(), padding='max_length', truncation=True, max_length=token_limit)
  train_negative_encodings = tokenizer(train_set['negative_text'].tolist(), padding='max_length', truncation=True, max_length=token_limit)
  
  train_dataset = CFClassifcationDataset(
    anchor_encodings=train_anchor_encodings,
    positive_encodings=train_positive_encodings,
    negative_encodings=train_negative_encodings,
    labels=train_set['label'].tolist(),
    triplet_sample_masks=train_set['triplet_sample_mask'].tolist()
  )

  base_dataset = torch.load(f"{base_base_path}/dataset.pt")
  
  aug_dataset = {
    "train": train_dataset,
    "valid": base_dataset['valid'],
    "test": base_dataset['test']
  }
  
  torch.save(aug_dataset, f"{aug_base_path}/dataset.pt")
  

