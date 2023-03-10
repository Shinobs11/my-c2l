import json, os, pickle, torch, logging, typing, numpy as np, glob, argparse, wandb
from re import L
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


from torchmetrics.classification import MulticlassAccuracy



torch.manual_seed(0)
import numpy as np
np.random.seed(0)
import random
random.seed(0)



handler = RotatingFileHandler(
    "logs/contrastiveLearning.log",
    maxBytes=10000000,  
)

log = logging.getLogger("contrastiveLearning")
log.setLevel(logging.DEBUG)
log.addHandler(handler)



def constrastiveTrain(
  dataset_name: str,
  batch_size: int,
  epoch_num: int,
  lambda_weight: float
):

  DATASET_NAME = dataset_name
  DATASET_PATH = f"datasets/{DATASET_NAME}/augmented_triplets"
  OUTPUT_PATH = f"checkpoints/{DATASET_NAME}/augmented_model"
  EPOCH_NUM = epoch_num
  BATCH_SIZE = batch_size







  train_set:pd.DataFrame = pload(os.path.join(DATASET_PATH, 'train_set'))[0:10]
  valid_set:pd.DataFrame = pload(os.path.join(DATASET_PATH, 'valid_set'))[0:10]


  train_labels = train_set['label'].to_list()
  valid_labels = valid_set['label'].to_list()

  
  anchor_train_texts = train_set['anchor_text'].to_list()
  anchor_valid_texts = valid_set['text'].to_list()

  positive_train_texts = train_set['positive_text'].to_list()


  negative_train_texts = train_set['negative_text'].to_list()


  train_triplet_sample_masks = train_set['triplet_sample_mask'].to_list()



  tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
  anchor_train_encodings = tokenizer(anchor_train_texts, truncation=True, padding=True)
  anchor_valid_encodings = tokenizer(anchor_valid_texts, truncation=True, padding=True)

  positive_train_encodings = tokenizer(positive_train_texts, truncation=True, padding=True)


  negative_train_encodings = tokenizer(negative_train_texts, truncation=True, padding=True)


  train_dataset = CFClassifcationDataset(anchor_train_encodings, positive_train_encodings, negative_train_encodings, train_triplet_sample_masks, train_labels)
  valid_dataset = ClassificationDataset(anchor_valid_encodings, valid_labels)

  train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
  valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

  num_classes = -1
  if len(train_loader.dataset[0]['label'].shape) == 1:
    num_classes = train_loader.dataset[0]['label'].shape[0]
  else:
    print("Invalid label shape")
    exit()


  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  model:BertForCounterfactualRobustness = BertForCounterfactualRobustness.from_pretrained('bert-base-uncased', num_labels=num_classes) #type:ignore
  model.to(device)
  optim = torch.optim.AdamW(model.parameters(), lr=5e-5)
  warmup = get_constant_schedule_with_warmup(optim, num_warmup_steps=int(len(train_loader)*0.10))
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=3, verbose=True)
  best_epoch = -1
  best_acc = 0
  steps = 0
  all_loss = []
  for epoch in range(EPOCH_NUM):
    train_metrics: MulticlassAccuracy = MulticlassAccuracy(num_classes=num_classes).to(device)
    valid_metrics: MulticlassAccuracy = MulticlassAccuracy(num_classes=num_classes).to(device)

    model.train()
    train_progress_bar = tqdm(train_loader)
    for batch in train_progress_bar:
      optim.zero_grad()

      anchor_input_ids = batch['anchor_input_ids'].to(device, dtype=torch.long)
      anchor_attention_mask = batch['anchor_attention_mask'].to(device, dtype=torch.long)
      anchor_token_type_ids = batch['anchor_token_type_ids'].to(device, dtype=torch.long)

      positive_input_ids = batch['positive_input_ids'].to(device, dtype=torch.long)
      positive_attention_mask = batch['positive_attention_mask'].to(device, dtype=torch.long)
      positive_token_type_ids = batch['positive_token_type_ids'].to(device, dtype=torch.long)

      negative_input_ids = batch['negative_input_ids'].to(device, dtype=torch.long)
      negative_attention_mask= batch['negative_attention_mask'].to(device, dtype=torch.long)
      negative_token_type_ids = batch['negative_token_type_ids'].to(device, dtype=torch.long)
      
      triplet_sample_masks = batch['triplet_sample_mask'].to(device)

      labels = batch['label'].to(device, dtype=torch.float) 


      true_labels = labels
      # _, labels = torch.max(labels, dim = 1) 


   
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
        labels=labels)

      loss = outputs[0]
      logits = outputs[1]

      _, true_labels_idx = true_labels.max(dim=1)
      _, pred_labels_idx = logits.max(dim=1)

      train_metrics.update(pred_labels_idx, true_labels_idx)


      loss.sum().backward()

      train_progress_bar.set_description("Epoch train accuracy: %f" % train_metrics.compute().item())
      optim.step()
      warmup.step()
      # scheduler.step()
      steps += 1

    # print(f"Total Train Accuracy: {total_train_accuracy}")

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
                    anchor_token_type_ids = token_type_ids
                    )
            loss = outputs[0]
            logits = outputs[1]
            val_loss += loss.sum()
            _, pred_labels_idx = logits.max(dim=1)
            _, true_labels_idx = true_labels.max(dim=1)

            log.debug(f"pred_labels_idx: {pred_labels_idx} true_labels_idx: {true_labels_idx}\n")

            valid_metrics.update(pred_labels_idx, true_labels_idx)
    scheduler.step(val_loss)
    accuracy = valid_metrics.compute().item()
    log.debug(f"Validation accuracy for epoch {epoch}: {accuracy}")
    wandb.log({"accuracy": accuracy, "valid_loss": val_loss})
    if accuracy >= best_acc:
        best_epoch = epoch
        best_acc = accuracy
    print(f"Accuracy: {accuracy}")
    mainLogger.info(f"contrastiveLearning epoch {epoch} finished with {accuracy} accuracy.")
    model.save_pretrained(os.path.join(OUTPUT_PATH, f"epoch_{epoch}")) #type:ignore
  pdump(all_loss, os.path.join(OUTPUT_PATH,"training_loss"))
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




### Configuring misc. stuff
from transformers import logging
logging.set_verbosity_error()
