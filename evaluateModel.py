from ast import Mult
import json, os, pickle, torch, logging, typing, numpy as np, glob, argparse, wandb
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers.models.bert.modeling_bert import SequenceClassifierOutput
from src.classes.datasets import ClassificationDataset
from src.utils.pickleUtils import pdump, pload
from src.proecssing import correct_count
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy
from tqdm import tqdm
import logging
from src.logging.logSetup import logSetup, mainLogger
from src.classes.model import BertForCounterfactualRobustness


torch.manual_seed(0)
import numpy as np
np.random.seed(0)
import random
random.seed(0)



log_path = os.path.join(os.path.split(str(__file__))[0], "logs", "eval")
log_info_path = os.path.join(log_path, "eval_info.log")
log_error_path = os.path.join(log_path, "eval_error.log")
info_logger = logSetup(
  logger_name="eval_info",
  logger_path=log_info_path,
  level=logging.INFO,
  format=None
)





def evaluateModel(
  dataset_name: str,
  batch_size: int,
  use_cl_model: bool = False
):
  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  DATASET_NAME = dataset_name
  DATASET_PATH = f"datasets/{dataset_name}/augmented_triplets"
  BATCH_SIZE = batch_size

  if not use_cl_model:
    MODEL_PATH = f"checkpoints/{DATASET_NAME}/model"
  else:
    MODEL_PATH = f"checkpoints/{DATASET_NAME}/augmented_model"

  def loadTestData():
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    test_set = pload(os.path.join(DATASET_PATH, 'test_set'))
    test_texts = test_set['text'].tolist()
    test_labels = test_set['label'].tolist()
    test_encodings = tokenizer(test_texts, padding=True, truncation=True)
    pdump(test_encodings, os.path.join(DATASET_PATH, 'test_encodings'))
    test_dataset = ClassificationDataset(labels=test_labels, encodings=test_encodings)
    test_loader = DataLoader(
      test_dataset,
      batch_size=BATCH_SIZE,
      shuffle=False,
      persistent_workers=False,
      pin_memory=False
    )
    return test_loader
  
  

  test_loader: DataLoader = loadTestData()
  num_labels = -1
  if len(test_loader.dataset[0]['label'].shape) == 1:
    num_labels = test_loader.dataset[0]['label'].shape[0]
  else:
    print("Invalid label shape")
    exit()

  torch.cuda.empty_cache()
  model: torch.nn.Module = BertForCounterfactualRobustness.from_pretrained(os.path.join(MODEL_PATH, "best_epoch"), num_labels=num_labels) #type:ignore
  model: torch.nn.Module = torch.compile(model) #type:ignore
  model.eval()
  model.to(device)




  metrics:MetricCollection = MetricCollection({
    "accuracy": MulticlassAccuracy(num_classes=num_labels)
  }
  ).to(device)



  with torch.no_grad():
    for batch in tqdm(test_loader):

      input_ids = batch["input_ids"].to(device)
      attention_mask = batch["attention_mask"].to(device)
      token_type_ids = batch["token_type_ids"].to(device)
      true_labels =  batch['label'].to(device)




      outputs = model.forward(
        anchor_input_ids=input_ids,
        anchor_attention_mask=attention_mask,
        anchor_token_type_ids=token_type_ids
      )

      logits = outputs[0]
    
      _, true_labels_idx = torch.max(true_labels, dim = 1)
      _, pred_labels_idx = torch.max(logits, dim = 1)
      info_logger.info(f"pred_labels_idx: {pred_labels_idx}")
      info_logger.info(f"true_labels_idx: {true_labels_idx}")

      

      metrics(preds=pred_labels_idx, target=true_labels_idx)

  model_type = "cl" if use_cl_model else "base"
  accuracy = metrics.compute()["accuracy"].item()
  wandb.log({"eval accuracy": accuracy})

  info_logger.info(f"{model_type} model evaluation accuracy: {accuracy}")
  mainLogger.info(f"{model_type} model evaluation accuracy: {accuracy}")