import json, os, pickle, torch, logging, gc, typing, numpy as np, glob, argparse, wandb
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from tqdm import tqdm
from transformers import BertTokenizerFast, BertForMaskedLM, BertForSequenceClassification, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
import transformers as T
from logging.handlers import RotatingFileHandler
from src.logging.logSetup import mainLogger
from src.utils.pickleUtils import pdump, pload, pjoin
from src.proecssing import correct_count
from transformers.models.bert.modeling_bert import SequenceClassifierOutput
from torch import Tensor
from  src.classes.datasets import CFClassifcationDataset, ClassificationDataset
import pandas as pd
from src.classes.model import BertForCounterfactualRobustness

from src.logging import log_memory
from torchmetrics.classification import MulticlassAccuracy
from torch.cuda.amp.autocast_mode import autocast








handler = RotatingFileHandler(
    "logs/contrastiveLearning.log",
    maxBytes=10000000,  
)

log = logging.getLogger("contrastiveLearning")
log.setLevel(logging.DEBUG)
log.addHandler(handler)









   
LOG_MEMORY_PATH = "logs/memory_log"




def constrastiveTrain(
  dataset_name: str,
  batch_size: int,
  epoch_num: int,
  lambda_weight: float
):
  
  torch.manual_seed(0)
  import numpy as np
  np.random.seed(0)
  import random
  random.seed(0)
  torch.cuda.manual_seed_all(0)
  torch.use_deterministic_algorithms(True)
  
  
  log_memory(LOG_MEMORY_PATH, "cl_before.json")
  
  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  if torch.cuda.is_available():
    torch.cuda.empty_cache()
  



  DATASET_NAME = dataset_name
  # DATASET_PATH = f"datasets/{DATASET_NAME}/augmented_triplets"
  DATASET_PATH = f"datasets/{DATASET_NAME}/base"
  OUTPUT_PATH = f"checkpoints/{DATASET_NAME}/augmented_model"
  EPOCH_NUM = epoch_num
  BATCH_SIZE = batch_size

  def loadDatasets():
    train_set:pd.DataFrame = pload(os.path.join(DATASET_PATH, 'train_set'))
    valid_set:pd.DataFrame = pload(os.path.join(DATASET_PATH, 'valid_set'))
    train_labels = train_set['label'].to_list()
    valid_labels = valid_set['label'].to_list()
    # anchor_train_texts = train_set['anchor_text'].to_list()
    anchor_train_texts = train_set['text'].to_list()
    anchor_valid_texts = valid_set['text'].to_list()
    # positive_train_texts = train_set['positive_text'].to_list()
    # negative_train_texts = train_set['negative_text'].to_list()
    # train_triplet_sample_masks = train_set['triplet_sample_mask'].to_list()
    train_triplet_sample_masks = [True for x in range(len(train_set))]
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    anchor_train_encodings = tokenizer(anchor_train_texts, truncation=True, padding=True)
    anchor_valid_encodings = tokenizer(anchor_valid_texts, truncation=True, padding=True)
    # positive_train_encodings = tokenizer(positive_train_texts, truncation=True, padding=True)
    # negative_train_encodings = tokenizer(negative_train_texts, truncation=True, padding=True)
    train_dataset = CFClassifcationDataset(anchor_train_encodings, anchor_train_encodings, anchor_train_encodings, train_triplet_sample_masks, train_labels)

    # train_dataset = CFClassifcationDataset(anchor_train_encodings, positive_train_encodings, negative_train_encodings, train_triplet_sample_masks, train_labels)
    valid_dataset = ClassificationDataset(anchor_valid_encodings, valid_labels)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    del train_set
    del valid_set
    del tokenizer
    return train_loader, valid_loader

  train_loader, valid_loader = loadDatasets()






 


  num_classes = -1
  if len(train_loader.dataset[0]["label"].shape) == 1: #type:ignore
    num_classes = train_loader.dataset[0]["label"].shape[0] #type:ignore
  elif len(train_loader.dataset[0]["label"].shape) == 2: #type:ignore
    num_classes = train_loader.dataset[0]["label"].shape[1]
  else:
    s = set()
    [s.add(x["label"]) for x in train_loader.dataset] #type:ignore
    num_classes = len(s)

  mainLogger.info(f"cl_num_classes: {num_classes}")

  torch.cuda.empty_cache()
  model:torch.nn.Module = BertForCounterfactualRobustness.from_pretrained('bert-base-uncased', num_labels=num_classes).to(device) #type:ignore
  # model:BertForCounterfactualRobustness = torch.compile(model)  #type:ignore
  
  optim = torch.optim.AdamW(model.parameters(), lr=5e-5)
  warmup = get_constant_schedule_with_warmup(optim, num_warmup_steps=int(len(train_loader)*0.25))
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=3, verbose=True)
  best_epoch = -1
  best_acc = 0
  steps = 0
  cumulative_training_loss = 0.0
  
  for epoch in range(EPOCH_NUM):
    epoch_training_loss = 0.0
    train_metrics: MulticlassAccuracy = MulticlassAccuracy(num_classes=num_classes).to(device)
    valid_metrics: MulticlassAccuracy = MulticlassAccuracy(num_classes=num_classes).to(device)

    model.train()
    train_progress_bar = tqdm(train_loader)
    for batch in train_progress_bar:
      optim.zero_grad()
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

      # labels = torch.argmax(labels, dim=1)
      true_labels = labels


      

      outputs = model.forward(
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

      loss = outputs['loss']
      logits = outputs['logits']

      _, true_labels_idx = true_labels.max(dim=1)
      _, pred_labels_idx = logits.max(dim=1)

      train_metrics.update(pred_labels_idx, true_labels_idx)


      training_loss = loss.sum()
      training_loss.backward()
      
      epoch_training_loss += training_loss
      train_progress_bar.set_description("Epoch train accuracy: %f" % train_metrics.compute().item())
      optim.step()
      warmup.step()
      # scheduler.step()
      steps += 1

    # print(f"Total Train Accuracy: {total_train_accuracy}")
    cumulative_training_loss += epoch_training_loss
    model.eval()
    accuracy = 0.0
    val_loss = 0.0
    for batch in valid_loader:
      with torch.no_grad():
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        true_labels = batch['label'].to(device)
        outputs = model(
                labels=true_labels,
                anchor_input_ids = input_ids,
                anchor_attention_mask = attention_mask,
                anchor_token_type_ids = token_type_ids,
                return_dict=True
                )
        loss = outputs['loss']
        logits = outputs['logits']
        val_loss += loss.sum()
        _, pred_labels_idx = logits.max(dim=1)
        _, true_labels_idx = true_labels.max(dim=1)

        valid_metrics.update(pred_labels_idx, true_labels_idx)
      accuracy = valid_metrics.compute().item()
        

        
    scheduler.step(val_loss)
    wandb.log({
      "cl_valid_accuracy": accuracy,
      "cl_valid_loss": val_loss,
      "cl_cumulative_training_loss": cumulative_training_loss,
      "cl_epoch_training_loss": epoch_training_loss,
      "cl_cumulative_accuracy": train_metrics.compute().item()
      })
    log.debug(f"Validation accuracy for epoch {epoch}: {accuracy}")
    
    if accuracy >= best_acc:
        best_epoch = epoch
        best_acc = accuracy
    print(f"Accuracy: {accuracy}")
    mainLogger.info(f"contrastiveLearning epoch {epoch} finished with {accuracy} accuracy.")
    model.save_pretrained(os.path.join(OUTPUT_PATH, f"epoch_{epoch}")) #type:ignore
  print(f"\nBest Model is epoch {best_epoch}.")
  print(f"\nBest accuracy is {best_acc}")
  mainLogger.info(f"contrastiveLearning finished with best epoch being {best_epoch} with {best_acc} accuracy.")
  
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




  # * Log memory before cleaning up *
  log_memory(LOG_MEMORY_PATH, "cl_after.json")

  # * Clean up memory to prevent OOM errors on next run *
  import torch._dynamo as dyn
  dyn.reset()
  del model, scheduler, warmup, optim
  dyn.reset()
  gc.collect()
  torch.cuda.empty_cache()


### Configuring misc. stuff
from transformers import logging
logging.set_verbosity_error()
