import json, os, pickle, torch, logging, typing, numpy as np
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from tqdm import tqdm
from transformers import BertTokenizerFast, BertForMaskedLM, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import transformers as T
from logging.handlers import RotatingFileHandler
from pickleUtils import pdump, pload, pjoin
from proecssing import correct_count
from transformers.models.bert.modeling_bert import SequenceClassifierOutput
from datasetClasses import DataItem
from torch import Tensor
from  datasetClasses import IMDBDataset
import pickle
import pandas as pd
DATASET_NAME = "imdb"
DATASET_PATH = f"./datasets/{DATASET_NAME}/base"
OUTPUT_PATH = f"checkpoints/{DATASET_NAME}/model"
TOPK_NUM = 4


import json, psutil
env = {}
with open("./env.json", mode="r") as f:
  env = json.load(f)


memAvailable = psutil.virtual_memory().available
estimatedMemConsumed = os.path.getsize(os.path.join(DATASET_PATH, "train_set.pickle.blosc")) * 3
USE_PINNED_MEMORY = True if (env['USE_PINNED_MEMORY'] & (memAvailable > estimatedMemConsumed)) == 1 else False
PRETRAIN_EPOCH_NUM = env['PRETRAIN_EPOCH_NUM']


def loadTrainData():


  tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
  train_set = pload(os.path.join(DATASET_PATH, "train_set"))
  train_texts:list[str] = train_set['review'].tolist()
  train_labels:list = train_set['sentiment'].tolist()
  train_encodings = tokenizer(train_texts, padding=True, truncation=True)
  pdump(train_encodings, os.path.join(DATASET_PATH, "train_encodings"))
  
  train_dataset = IMDBDataset(labels=train_labels, encodings=train_encodings)
  train_loader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=False,
    persistent_workers=False, ##Switch to true if dataloader is used multiple times
    pin_memory=False)
  return train_loader

def loadValData():
  tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

  val_set = pload(os.path.join(DATASET_PATH, "val_set"))
  val_texts:list[str] = val_set['review'].tolist()
  val_labels: list = val_set['sentiment'].tolist()
  val_encodings = tokenizer(val_texts, padding=True, truncation=True)
  pdump(val_encodings, os.path.join(DATASET_PATH, "val_encodings"))

  val_dataset = IMDBDataset(labels=val_labels, encodings=val_encodings)
  val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    persistent_workers=False,
    pin_memory=False
  )

  return val_loader


def pretrainModel():
  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  
  train_loader: DataLoader = loadTrainData()
  val_loader: DataLoader = loadValData()

  torch.cuda.empty_cache()
  model:torch.nn.Module = BertForSequenceClassification.from_pretrained('bert-base-uncased')  # type: ignore
  # model:torch.nn.Module = torch.compile(model) #type: ignore
  model:BertForSequenceClassification = torch.nn.DataParallel(model)  # type: ignore
  model.to(device)

  optim = AdamW(model.parameters(), lr=5e-5)
  scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=50, num_training_steps=len(train_loader)*PRETRAIN_EPOCH_NUM)

  steps = 0
  best_acc = 0
  best_epoch = -1
  all_loss = []
  total_train_accuracy = 0.0


  for epoch in range(PRETRAIN_EPOCH_NUM):
    epoch_loss = []
    model.train() # switch to training mode
    train_progress_bar = tqdm(train_loader)
    train_cor_count = 0
    train_total_size = 0
    epoch_steps = 0
    epoch_train_accuracy = 0.0
    for batch in train_progress_bar:
      optim.zero_grad()
      input_ids = batch['input_ids'].to(device)
      attention_mask = batch['attention_mask'].to(device)
      token_type_ids = batch['token_type_ids'].to(device)
      labels = batch['labels'].to(device)
      original_labels = labels
      _, labels = torch.max(labels, dim=1)


      outputs:SequenceClassifierOutput|tuple[torch.Tensor] = model.forward(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        labels=labels
      )
      assert isinstance(outputs, SequenceClassifierOutput)


      loss:torch.Tensor = outputs['loss']
      logits:torch.Tensor = outputs['logits']


      

      train_cor_count += correct_count(logits, original_labels)
      train_total_size  += len(original_labels)
      batch_accuracy = train_cor_count/len(original_labels)
      epoch_train_accuracy += batch_accuracy


      loss.sum().backward()
      epoch_loss.append(loss.sum())
      train_progress_bar.set_description("Batch Accuracy %f" % batch_accuracy)
      optim.step()
      scheduler.step()
      epoch_steps+=1
      steps+=1
    epoch_train_accuracy = epoch_train_accuracy/epoch_steps
    print(f"Epoch Accuracy: {epoch_train_accuracy}")
    all_loss.append(epoch_loss)
    model.eval()# setting model to evaluation mode

    cor_count = 0
    total_size = 0
    accuracy = 0.0

    for batch in val_loader:
      with torch.no_grad():
          input_ids = batch['input_ids'].to(device)
          attention_mask = batch['attention_mask'].to(device)
          token_type_ids = batch['token_type_ids'].to(device)
          labels = batch['labels'].to(device)
          outputs:SequenceClassifierOutput|tuple[torch.Tensor] = model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
          )
          assert isinstance(outputs, SequenceClassifierOutput)
          logits = outputs['logits']
          cor_count += correct_count(logits, labels)
          total_size += len(labels)

    if total_size:
      accuracy = cor_count*1.0/total_size
    if accuracy >= best_acc:
      best_epoch = epoch
    print(f"Accuracy: {accuracy}")
    model.module.save_pretrained(os.path.join(OUTPUT_PATH, f"epoch_{epoch}")) #type:ignore
  pdump(all_loss, os.path.join(OUTPUT_PATH,"training_loss"))
  print(f"\nBest model is epoch {best_epoch}.")
  os.rmdir(os.path.join(OUTPUT_PATH, "best_epoch"))
  os.rename(os.path.join(OUTPUT_PATH, f"epoch_{best_epoch}"), os.path.join(OUTPUT_PATH, "best_epoch"))


pretrainModel()