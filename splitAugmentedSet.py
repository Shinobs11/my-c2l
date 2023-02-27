import json, os, pandas as pd
from src.utils.pickleUtils import pload, pdump
def splitAugmentedSet(dataset_path:str):
  aug_set_file = open(dataset_path, mode='r')
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
  
  train_set = aug_df[0:70]
