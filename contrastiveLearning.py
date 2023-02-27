import json, os, pickle, torch, logging, typing, numpy as np, glob, argparse
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from tqdm import tqdm
from transformers import BertTokenizerFast, BertForMaskedLM, BertForSequenceClassification, get_linear_schedule_with_warmup
import transformers as T
from logging.handlers import RotatingFileHandler

from src.utils.pickleUtils import pdump, pload, pjoin
from src.proecssing import correct_count
from transformers.models.bert.modeling_bert import SequenceClassifierOutput
from torch import Tensor
from  src.classes.datasets import CFClassifcationDataset, CFIMDbDataset, IMDBDataset
import pickle
import pandas as pd
from src.classes.model import BertForCounterfactualRobustness


from torchmetrics.classification import MulticlassAccuracy



import json

def constrastiveTrain(
  dataset_name: str,
  batch_size: int,
  epoch_num: int,
  use_margin_loss: bool,
  lambda_weight: float,
  use_cache: bool
):

  DATASET_NAME = dataset_name
  DATASET_PATH = f"datasets/{DATASET_NAME}/augmented_triplets"
  OUTPUT_PATH = f"checkpoints/{DATASET_NAME}/augmented_model"
  EPOCH_NUM = epoch_num
  TOPK_NUM = 4
  BATCH_SIZE = batch_size








  train_set:pd.DataFrame = pload(os.path.join(DATASET_PATH, 'train_set'))
  valid_set:pd.DataFrame = pload(os.path.join(DATASET_PATH, 'valid_set'))

  train_labels = train_set['label']
  valid_labels = valid_set['label']

  anchor_train_texts = train_set['anchor_text']
  anchor_valid_texts = valid_set['anchor_text']

  positive_train_texts = train_set['positive_text']
  positive_valid_texts = valid_set['positive_text']

  negative_train_texts = train_set['negative_text']
  negative_valid_texts = valid_set['negative_text']

  train_triplet_sample_masks = None
  valid_triplet_sample_masks = None

  if 'triplet_sample_mask' in train_set[0].keys():
    train_triplet_sample_masks = train_set['triplet_sample_mask']
  if 'triplet_sample_mask' in valid_set[0].keys():
    valid_triplet_sample_masks = valid_set['triplet_sample_mask']


  tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
  anchor_train_encodings = tokenizer(anchor_train_texts, truncation=True, padding=True)
  anchor_valid_encodings = tokenizer(anchor_valid_texts, truncation=True, padding=True)

  positive_train_encodings = tokenizer(positive_train_texts, truncation=True, padding=True)
  positive_valid_encodings = tokenizer(positive_valid_texts, truncation=True, padding=True)

  negative_train_encodings = tokenizer(negative_train_texts, truncation=True, padding=True)
  negative_valid_encodings = tokenizer(negative_valid_texts, truncation=True, padding=True)


  train_dataset = CFClassifcationDataset(anchor_train_encodings, positive_train_encodings, negative_train_encodings, train_triplet_sample_masks, train_labels)
  valid_dataset = CFClassifcationDataset(anchor_valid_encodings, positive_valid_encodings, negative_valid_encodings, valid_triplet_sample_masks, valid_labels)

  train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
  valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

  num_classes = -1
  if len(train_loader.dataset[0]["labels"].shape) == 1:
    num_classes = train_loader.dataset[0]["labels"].shape[0]
  else:
    print("Invalid label shape")
    exit()


  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  model:BertForCounterfactualRobustness = BertForCounterfactualRobustness.from_pretrained('bert-base-uncased', num_labels=num_classes) #type:ignore
  model.to(device)
  optim = torch.optim.AdamW(model.parameters(), lr=5e-5)
  scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=50, num_training_steps=len(train_loader) * EPOCH_NUM)

  best_epoch = -1
  best_acc = 0
  steps = 0
  all_loss = []
  for epoch in range(EPOCH_NUM):
    train_metrics: MulticlassAccuracy = MulticlassAccuracy(num_classes=num_classes)
    valid_metrics: MulticlassAccuracy = MulticlassAccuracy(num_classes=num_classes)

    epoch_loss = []
    model.train()
    train_progress_bar = tqdm(train_loader)
    for batch in train_progress_bar:
      optim.zero_grad()

      anchor_input_ids = batch['anchor_input_ids'].to(device)
      anchor_attention_mask = batch['anchor_attention_mask'].to(device)
      anchor_token_type_ids = batch['anchor_token_type_ids'].to(device)

      positive_input_ids = batch['positive_input_ids'].to(device)
      positive_attention_mask = batch['positive_attention_mask'].to(device)
      positive_token_type_ids = batch['positive_token_type_ids'].to(device)

      negative_input_ids = batch['negative_input_ids'].to(device)
      negative_attention_mask= batch['negative_attention_mask'].to(device)
      negative_token_type_ids = batch['negative_token_type_ids'].to(device)
      
      triplet_sample_masks = batch['triplet_sample_masks'].to(device)

      labels = batch['labels'].to(device)

      true_labels = labels
      # _, labels = torch.max(labels, dim = 1) 


      if use_margin_loss:
            #outputs = model(anc_input_ids, anc_attention_mask, pos_input_ids, pos_attention_mask, neg_input_ids, neg_attention_mask, labels=labels)
        outputs = model(
          anchor_input_ids, 
          anchor_attention_mask, 
          positive_input_ids, 
          positive_attention_mask, 
          negative_input_ids, 
          negative_attention_mask, 
          anchor_token_type_ids=anchor_token_type_ids,
          positive_token_type_ids=positive_token_type_ids,
          negative_token_type_ids=negative_token_type_ids,
          triplet_sample_masks=triplet_sample_masks, 
          lambda_weight=lambda_weight, 
          labels=labels)
      else:
        outputs = model(
          anchor_input_ids,
          anchor_attention_mask ,
          anchor_token_type_ids=anchor_token_type_ids,
          labels=labels)
      loss = outputs[0]
      logits = outputs[1]

      _, true_labels_idx = true_labels.max(dim=1)
      _, pred_labels_idx = logits.max(dim=1)

      train_metrics.update(pred_labels_idx, true_labels_idx)


      loss.sum().backward()

      train_progress_bar.set_description("Epoch train accuracy: %f" % train_metrics.compute().item())
      optim.step()
      scheduler.step()
      steps += 1

    # print(f"Total Train Accuracy: {total_train_accuracy}")

    model.eval()
    for batch in valid_loader:
        with torch.no_grad():
            anc_input_ids = batch['anchor_input_ids'].to(device)
            anc_attention_mask = batch['anchor_attention_mask'].to(device)
            anc_token_type_ids = batch['anchor_token_type_ids'].to(device)
            true_labels = batch['labels'].to(device)
            outputs = model(
                    anc_input_ids,
                    anc_attention_mask,
                    anchor_token_type_ids=anc_token_type_ids)

            logits = outputs[0]
            _, pred_labels_idx = logits.max(dim=1)
            _, true_labels_idx = true_labels.max(dim=1)
            valid_metrics.update(pred_labels_idx, true_labels_idx)

    accuracy = valid_metrics.compute().item()
    if accuracy >= best_acc:
        best_epoch = epoch
        best_acc = accuracy
    print(f"Accuracy: {accuracy}")
    model.save_pretrained(os.path.join(OUTPUT_PATH, f"epoch_{epoch}")) #type:ignore
  pdump(all_loss, os.path.join(OUTPUT_PATH,"training_loss"))
  print(f"\nBest Model is epoch {best_epoch}.")
  print(f"\nBest accuracy is {best_acc}")
  try:
      for x in glob.glob(os.path.join(OUTPUT_PATH, "best_epoch", "*")):
        os.remove(x)
      os.rmdir(os.path.join(OUTPUT_PATH, "best_epoch"))
  except:
      pass
  os.rename(os.path.join(OUTPUT_PATH, f"epoch_{best_epoch}"), os.path.join(OUTPUT_PATH, "best_epoch"))
  try:
      for x in glob.glob(os.path.join(OUTPUT_PATH, "epoch_*")):
        for y in glob.glob(os.path.join(x, "*")):
          os.remove(y)
        os.rmdir(x)
  except:
      pass
