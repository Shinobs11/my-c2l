import argparse
import gc
import glob
import json
import logging
import os
import pickle
import typing
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import psutil
import torch
from torch import amp
from torch.cuda.amp.autocast_mode import autocast
from torch.utils.data import DataLoader, Dataset
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy
from tqdm import tqdm
from transformers import (BertForSequenceClassification, BertTokenizerFast,
                          get_constant_schedule_with_warmup,
                          get_linear_schedule_with_warmup)
from transformers.models.bert.modeling_bert import SequenceClassifierOutput
from src.logging.logSetup import mainLogger
import wandb
from src.classes.datasets import ClassificationDataset
from src.classes.model import BertForCounterfactualRobustness
from src.logging import log_memory
from src.proecssing import correct_count
from src.utils.pickleUtils import pdump, pload



def pretrainBERT(
  dataset_name: str,
  batch_size: int,
  epoch_num: int,
  use_pinned_memory: bool,
  lr: float,
):
  torch.manual_seed(0)
  import numpy as np
  np.random.seed(0)
  import random
  random.seed(0)
  torch.cuda.manual_seed_all(0)
  torch.use_deterministic_algorithms(True)

  DATASET_NAME = dataset_name
  BATCH_SIZE = batch_size
  EPOCH_NUM = epoch_num


  DATASET_PATH = f"datasets/{DATASET_NAME}/base"
  OUTPUT_PATH = f"checkpoints/{DATASET_NAME}/model"


  LOG_MEMORY_PATH = "logs/memory_log"

  log_memory(LOG_MEMORY_PATH, "pt_before.json")







    

  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  if torch.cuda.is_available():
    torch.cuda.empty_cache()
  

  num_classes = -1
  if len(train_loader.dataset[0]["label"].shape) == 1: #type:ignore
    num_classes = train_loader.dataset[0]["label"].shape[0] #type:ignore
  elif len(train_loader.dataset[0]["label"].shape) == 2: #type:ignore
    num_classes = train_loader.dataset[0]["label"].shape[1]
  else:
    s = set()
    [s.add(x["label"]) for x in train_loader.dataset] #type:ignore
    num_classes = len(s)

  mainLogger.info(f"pt_num_classes: {num_classes}")

  torch.cuda.empty_cache()
  model:torch.nn.Module = BertForCounterfactualRobustness.from_pretrained('bert-base-uncased', num_labels=num_classes).to(device)  # type: ignore
  optim = torch.optim.AdamW(model.parameters(), lr=5e-5)
  warmup = get_linear_schedule_with_warmup(optim, num_warmup_steps=50, num_training_steps=len(train_loader)*EPOCH_NUM)
  reduce_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=3, verbose=True)

  steps = 0
  best_acc = 0
  best_epoch = -1
  cumulative_train_loss = 0.0

  for epoch in range(EPOCH_NUM):
    epoch_train_loss = 0.0
    train_metrics: MulticlassAccuracy  = MulticlassAccuracy(num_classes=num_classes).to(device)
    valid_metrics: MulticlassAccuracy = MulticlassAccuracy(num_classes=num_classes).to(device)
    model.train()
    train_progress_bar = tqdm(train_loader)

    for batch in train_progress_bar:
      torch.manual_seed(0)
      import numpy as np
      np.random.seed(0)
      import random
      random.seed(0)
      torch.cuda.manual_seed_all(0)
      torch.use_deterministic_algorithms(True)
      
      optim.zero_grad()
      
      input_ids = batch['input_ids'].to(device, dtype=torch.int)
      attention_mask = batch['attention_mask'].to(device, dtype=torch.int)
      token_type_ids = batch['token_type_ids'].to(device, dtype=torch.int)
      labels = batch['label'].to(device)


      true_labels = labels #shape = [n_labels]

  

      outputs:Union[SequenceClassifierOutput, Tuple[torch.Tensor]] = model.forward(
        anchor_input_ids=input_ids,
        anchor_attention_mask=attention_mask,
        anchor_token_type_ids=token_type_ids,
        labels=labels,
        return_dict=True
      )
      assert isinstance(outputs, SequenceClassifierOutput)
      loss:torch.Tensor = outputs['loss']
      logits:torch.Tensor = outputs['logits']

    
      _, pred_labels_idx = logits.max(dim=1)
      _, true_labels_idx = true_labels.max(dim=1)

      train_metrics.update(pred_labels_idx, true_labels_idx)

      training_loss = loss.sum()
      epoch_train_loss += training_loss
      training_loss.backward()
      
      train_progress_bar.set_description("Epoch train_accuracy %f" % train_metrics.compute().item())
      optim.step()
      warmup.step()
      steps+=1
      
    cumulative_train_loss += epoch_train_loss
    model.eval()

    accuracy = 0.0
    val_loss = 0.0
    for batch in valid_loader:
      with torch.no_grad():
        input_ids = batch['input_ids'].to(device, dtype=torch.int)
        attention_mask = batch['attention_mask'].to(device, dtype=torch.int)
        token_type_ids = batch['token_type_ids'].to(device, dtype=torch.int)
        true_labels = batch['label'].to(device)

        outputs:Union[SequenceClassifierOutput, Tuple[torch.Tensor]] = model.forward(
          labels=true_labels,
          anchor_input_ids=input_ids,
          anchor_attention_mask=attention_mask,
          anchor_token_type_ids=token_type_ids,
          return_dict=True
        )
        assert isinstance(outputs, SequenceClassifierOutput)
        loss = outputs['loss']
        logits = outputs['logits']
        val_loss += loss.sum()
        _, pred_labels_idx = logits.max(dim=1)
        _, true_labels_idx = true_labels.max(dim=1)
        
        valid_metrics.update(pred_labels_idx, true_labels_idx)
      accuracy = valid_metrics.compute().item()

    reduce_lr.step(val_loss)
    wandb.log({
      "base_valid_accuracy": accuracy,
      "base_valid_loss": val_loss,
      "base_cumulative_train_loss": cumulative_train_loss,
      "base_epoch_train_loss": epoch_train_loss,
      "base_epoch_train_accuracy": train_metrics.compute().item()
      })

    if accuracy >= best_acc:
      best_epoch = epoch
      best_acc = accuracy
    print(f"Accuracy: {accuracy}")
    
    model.save_pretrained(os.path.join(OUTPUT_PATH, f"epoch_{epoch}")) #type:ignore
  print(f"\nBest model is epoch {best_epoch}.")
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


  # * Clean up memory to prevent OOM errors on next run *
  import torch._dynamo as dyn
  log_memory(LOG_MEMORY_PATH, "pt_after.json")
  dyn.reset()
  del model, reduce_lr, optim, warmup
  dyn.reset()
  gc.collect()
  torch.cuda.empty_cache()
        
      






### Configuring misc. stuff
from transformers import logging
logging.set_verbosity_error()
