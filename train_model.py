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
from src.configs.bert_config import bert_config

def set_determinism(seed: int):
  torch.manual_seed(seed)
  import numpy as np
  np.random.seed(seed)
  import random
  random.seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.use_deterministic_algorithms(True)


def load_datasets(dataset_name: str, use_cl: bool):
  DATASET_PATH = f"datasets/{dataset_name}/{'cl' if use_cl else 'base'}"
  datasets = torch.load(f"{DATASET_PATH}/dataset.pt")
  train_dataset = datasets['train']
  valid_dataset = datasets['valid']
  return train_dataset, valid_dataset

def load_model(lr: float, num_classes: int, train_size: int, epoch_num: int,  device: torch.device):
  model:torch.nn.Module = BertForCounterfactualRobustness.from_pretrained('bert-base-uncased', ignore_mismatched_sizes=True, config=bert_config).to(device)  # type: ignore
  optim = torch.optim.AdamW(model.parameters(), lr=lr)
  warmup = get_linear_schedule_with_warmup(optim, num_warmup_steps=50, num_training_steps=train_size*epoch_num)
  reduce_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=3, verbose=True)
  return model, optim, warmup, reduce_lr

def train_model(
  dataset_name: str,
  batch_size: int,
  epoch_num: int,
  use_pinned_memory: bool,
  lr: float,
  lambda_weight: float,
  use_cl: bool = False,
  use_wandb: bool = True,
):

  set_determinism(0)

  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  
  OUTPUT_PATH = f"checkpoints/{dataset_name}/model/{'cl' if use_cl else 'base'}"
  
  
  train_dataset, valid_dataset = load_datasets(dataset_name, use_cl)
  
  metadata = torch.load(f"datasets/{dataset_name}/dataset_meta.pt")
  num_classes = metadata['num_classes']
  train_size = metadata['train_size']

  model, optim, warmup, reduce_lr = load_model(lr, num_classes, train_size, epoch_num, device)

  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, pin_memory=use_pinned_memory)
  valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, pin_memory=use_pinned_memory)
  
  
  def train_loop_base(batch) -> tuple[torch.Tensor, Union[SequenceClassifierOutput, Tuple[torch.Tensor]]]:
    input_ids = batch['input_ids'].to(device, dtype=torch.int)
    attention_mask = batch['attention_mask'].to(device, dtype=torch.int)
    token_type_ids = batch['token_type_ids'].to(device, dtype=torch.int)
    labels = batch['label'].to(device)
    return labels, model.forward(
        anchor_input_ids=input_ids,
        anchor_attention_mask=attention_mask,
        anchor_token_type_ids=token_type_ids,
        labels=labels,
        return_dict=True
      )
  
  def train_loop_cl(batch) -> tuple[torch.Tensor, Union[SequenceClassifierOutput, Tuple[torch.Tensor]]]:
    anchor_input_ids = batch['anchor_input_ids'].to(device, dtype=torch.int)
    anchor_attention_mask = batch['anchor_attention_mask'].to(device, dtype=torch.int)
    anchor_token_type_ids = batch['anchor_token_type_ids'].to(device, dtype=torch.int)

    positive_input_ids = batch['positive_input_ids'].to(device, dtype=torch.int)
    positive_attention_mask = batch['positive_attention_mask'].to(device, dtype=torch.int)
    positive_token_type_ids = batch['positive_token_type_ids'].to(device, dtype=torch.int)

    negative_input_ids = batch['negative_input_ids'].to(device, dtype=torch.int)
    negative_attention_mask= batch['negative_attention_mask'].to(device, dtype=torch.int)
    negative_token_type_ids = batch['negative_token_type_ids'].to(device, dtype=torch.int)
    
    triplet_sample_masks = batch['triplet_sample_mask'].to(device)

    labels = batch['label'].to(device) 
    return labels, model.forward(
        anchor_input_ids=anchor_input_ids, 
        anchor_attention_mask=anchor_attention_mask, 
        positive_input_ids = positive_input_ids, 
        positive_attention_mask= positive_attention_mask, 
        negative_input_ids= negative_input_ids, 
        negative_attention_mask= negative_attention_mask, 
        anchor_token_type_ids=anchor_token_type_ids,
        positive_token_type_ids=positive_token_type_ids,
        negative_token_type_ids=negative_token_type_ids,
        triplet_sample_masks=triplet_sample_masks, 
        lambda_weight=lambda_weight, 
        labels=labels,
        return_dict=True
        )
    
  
  steps = 0
  best_acc = 0
  best_epoch = -1
  cumulative_train_loss = 0.0

  for epoch in range(epoch_num):
    epoch_train_loss = 0.0
    train_metrics: MulticlassAccuracy  = MulticlassAccuracy(num_classes=num_classes).to(device)
    valid_metrics: MulticlassAccuracy = MulticlassAccuracy(num_classes=num_classes).to(device)
    model.train()
    train_progress_bar = tqdm(train_loader)

    for batch in train_progress_bar:
      set_determinism(0)
      
      optim.zero_grad()
      
      
      if use_cl:
        labels, outputs = train_loop_cl(batch)  
      else:
        labels, outputs = train_loop_base(batch)
        
      assert isinstance(outputs, SequenceClassifierOutput)
      loss:torch.Tensor = outputs['loss']
      logits:torch.Tensor = outputs['logits']

      pred_labels_idx = logits.argmax(dim=1)
      true_labels_idx = labels.argmax(dim=1)

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
    if use_wandb:
      wandb.log({
        f"{'cl' if use_cl else 'base'}_valid_accuracy": accuracy,
        f"{'cl' if use_cl else 'base'}_valid_loss": val_loss,
        f"{'cl' if use_cl else 'base'}_cumulative_train_loss": cumulative_train_loss,
        f"{'cl' if use_cl else 'base'}_epoch_train_loss": epoch_train_loss,
        f"{'cl' if use_cl else 'base'}_epoch_train_accuracy": train_metrics.compute().item()
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
  # log_memory(LOG_MEMORY_PATH, "pt_after.json")
  dyn.reset()
  del model, reduce_lr, optim, warmup
  dyn.reset()
  gc.collect()
  torch.cuda.empty_cache()
        
      






### Configuring misc. stuff
from transformers import logging
logging.set_verbosity_error()
