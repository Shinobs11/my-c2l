import sys, os
sys.path.append(f"{os.getcwd()}")
from operator import index
import pandas as pd
import numpy as np
from src.utils.pickleUtils import pdump, pload
import os
from transformers import BertTokenizerFast
from src.classes.datasets import ClassificationDataset
#for CHI, review being fake is Y or 0, N or 1
import torch
def split_dataset(train_size: int = 300, test_size: int = 100, token_limit: int = 512):
  tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
  yelp_chi_train = pd.read_csv("datasets/yelp_chi/yelp_chi_train.csv", index_col = 0).reset_index(drop=True)
  yelp_chi_test = pd.read_csv("datasets/yelp_chi/yelp_chi_test.csv", index_col = 0).reset_index(drop=True)
  yelp_nyc_train = pd.read_csv("datasets/yelp_nyc/yelp_nyc_train.csv", index_col = 0).reset_index(drop=True)
  yelp_nyc_test = pd.read_csv("datasets/yelp_nyc/yelp_nyc_test.csv", index_col = 0).reset_index(drop=True)
  yelp_zip_train = pd.read_csv("datasets/yelp_zip/yelp_zip_train.csv", index_col = 0).reset_index(drop=True)
  yelp_zip_test = pd.read_csv("datasets/yelp_zip/yelp_zip_test.csv", index_col = 0).reset_index(drop=True)


  yelp_chi_train = yelp_chi_train.drop(columns=["review_id", "reviewer_id", "product_id", "date", "star", "label"])
  yelp_chi_test = yelp_chi_test.drop(columns=["review_id", "reviewer_id", "product_id", "date", "star", "label"])
  yelp_nyc_train = yelp_nyc_train.drop(columns=["review_id", "business_id", "date", "stars", "label"])
  yelp_nyc_test = yelp_nyc_test.drop(columns=["review_id", "business_id", "date", "stars", "label"])
  yelp_zip_train = yelp_zip_train.drop(columns=["review_id", "business_id", "date", "stars", "label"])
  yelp_zip_test = yelp_zip_test.drop(columns=["review_id", "business_id", "date", "stars", "label"])

  yelps = [yelp_chi_train, yelp_chi_test, yelp_nyc_train, yelp_nyc_test, yelp_zip_train, yelp_zip_test]
  
  for yelp in yelps:
    yelp.rename(columns={"content": "text", "label_new": "label"}, inplace=True)
    yelp["label"] = yelp["label"].transform((lambda x: [0.0, 1.0] if x == 1 else [1.0, 0.0]))
    yelp_tokens = tokenizer(yelp["text"].to_list(), return_length=True, return_attention_mask=False, return_token_type_ids=False)
    yelp['length'] = yelp_tokens['length']
    yelp.drop(yelp[yelp['length'] > token_limit].index, inplace=True)
    yelp.reset_index(drop=True, inplace=True)
  
  
  
  yelp_train = [yelp_chi_train, yelp_nyc_train, yelp_zip_train]
  yelp_test = [yelp_chi_test, yelp_nyc_test, yelp_zip_test]
  
  for train, test in zip(yelp_train, yelp_test):
    train.drop(train.index[train_size:], inplace=True)
    test.drop(test.index[test_size:], inplace=True)
  
  
  chi_train_set = yelp_chi_train.drop(yelp_chi_train.index[int(len(yelp_chi_train)*0.75):])
  chi_valid_set = yelp_chi_train.drop(yelp_chi_train.index[:int(len(yelp_chi_train)*0.75)])
  chi_test_set = yelp_chi_test

  nyc_train_set = yelp_nyc_train.drop(yelp_nyc_train.index[int(len(yelp_nyc_train)*0.75):])
  nyc_valid_set = yelp_nyc_train.drop(yelp_nyc_train.index[:int(len(yelp_nyc_train)*0.75)])
  nyc_test_set = yelp_nyc_test

  zip_train_set = yelp_zip_train.drop(yelp_zip_train.index[int(len(yelp_zip_train)*0.75):])
  zip_valid_set = yelp_zip_train.drop(yelp_zip_train.index[:int(len(yelp_zip_train)*0.75)])
  zip_test_set = yelp_zip_test



  CHI_PATH = "datasets/yelp_chi"
  NYC_PATH = "datasets/yelp_nyc"
  ZIP_PATH = "datasets/yelp_zip"


  for p in (CHI_PATH, NYC_PATH, ZIP_PATH):
    if not os.path.exists(os.path.join(p, "base")):
      os.makedirs(os.path.join(p, "base"))

  data_sets = (
    (chi_train_set, chi_valid_set, chi_test_set, CHI_PATH),
    (nyc_train_set, nyc_valid_set, nyc_test_set, NYC_PATH),
    (zip_train_set, zip_valid_set, zip_test_set, ZIP_PATH)
  )


  

  import csv
  for ds in data_sets:
    ds[0]['text'].transform((lambda x: x.strip('\"')))
    ds[1]['text'].transform((lambda x: x.strip('\"')))
    ds[2]['text'].transform((lambda x: x.strip('\"')))
    ds[0].to_csv(os.path.join(ds[3], "base", "train_set.csv"), quoting=csv.QUOTE_ALL, index=False)
    ds[1].to_csv(os.path.join(ds[3], "base", "valid_set.csv"), quoting=csv.QUOTE_ALL, index=False)
    ds[2].to_csv(os.path.join(ds[3], "base","test_set.csv"), quoting=csv.QUOTE_ALL, index=False)

    train_encodings = tokenizer(ds[0]['text'].tolist(), truncation=True, padding='max_length', max_length=token_limit)
    valid_encodings = tokenizer(ds[1]['text'].tolist(), truncation=True, padding='max_length', max_length=token_limit)
    test_encodings = tokenizer(ds[2]['text'].tolist(), truncation=True, padding='max_length', max_length=token_limit)
    train_labels = ds[0]['label'].tolist()
    valid_labels = ds[1]['label'].tolist()
    test_labels = ds[2]['label'].tolist()
    
    train_dataset = ClassificationDataset(train_encodings, train_labels)
    test_dataset = ClassificationDataset(test_encodings, test_labels)
    valid_dataset = ClassificationDataset(valid_encodings, valid_labels)
    
    
    
    datasets = {
      "train":train_dataset,
      "test":test_dataset,
      "valid":valid_dataset,
      "train_text": ds[0]['text'].tolist(),
      "test_text": ds[2]['text'].tolist(),
      "valid_text": ds[1]['text'].tolist()
      }
    dataset_meta = {
      "token_limit": token_limit,
      "train_size": train_size,
      "test_size": test_size,
      "num_classes": 2
    }
    torch.save(datasets, os.path.join(ds[3], "base", "dataset.pt"))
    torch.save(dataset_meta, os.path.join(ds[3], "dataset_meta.pt"))
    
    
    
    
    

