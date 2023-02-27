from ast import Mult
import json, os, pickle, torch, logging, typing, numpy as np, glob, argparse
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers.models.bert.modeling_bert import SequenceClassifierOutput
from src.classes.datasets import IMDBDataset
from src.utils.pickleUtils import pdump, pload
from src.proecssing import correct_count
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy
from tqdm import tqdm
import logging
from src.logging.logSetup import logSetup

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
  epoch_num: int,
  use_margin_loss: bool,
  lambda_weight: float,
  use_cache: bool
):
  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  DATASET_NAME = dataset_name
  DATASET_PATH = f"datasets/{dataset_name}/base"
  BATCH_SIZE = batch_size
  EPOCH_NUM = epoch_num
  USE_MARGIN_LOSS = use_margin_loss
  LAMBDA_WEIGHT = lambda_weight
  USE_ENCODING_CACHE = use_cache
  OUTPUT_PATH = f"checkpoints/{DATASET_NAME}/model"
  TOPK_NUM = 4
  # memAvailable = psutil.virtual_memory().available
  # estimatedMemConsumed = os.path.getsize(os.path.join(DATASET_PATH, "train_set.pickle.blosc")) * 3
  # USE_PINNED_MEMORY = True if (args.use_pinned_memory & (memAvailable > estimatedMemConsumed)) == 1 else False


  def loadTestData():
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    test_set = pload(os.path.join(DATASET_PATH, 'valid_set'))
    test_texts = test_set['text'].tolist()
    test_labels = test_set['label'].tolist()
    test_encodings = tokenizer(test_texts, padding=True, truncation=True)
    pdump(test_encodings, os.path.join(DATASET_PATH, 'test_encodings'))
    test_dataset = IMDBDataset(labels=test_labels, encodings=test_encodings)
    test_loader = DataLoader(
      test_dataset,
      batch_size=BATCH_SIZE,
      shuffle=True,
      persistent_workers=False,
      pin_memory=False
    )
    return test_loader
  
  

  test_loader: DataLoader = loadTestData()
  num_labels = -1
  if len(test_loader.dataset[0]["labels"].shape) == 1:
    num_labels = test_loader.dataset[0]["labels"].shape[0]
  else:
    print("Invalid label shape")
    exit()

  torch.cuda.empty_cache()
  model: torch.nn.Module = BertForSequenceClassification.from_pretrained(os.path.join(OUTPUT_PATH, "best_epoch"), num_labels=num_labels) #type: ignore
  # model: BertForSequenceClassification = torch.nn.DataParallel(model) #type:ignore #!only use w/ distributed
  model.eval()
  model.to(device)




  metrics:MetricCollection = MetricCollection([
    MulticlassAccuracy(num_classes=num_labels)
  ]
  ).to(device)



  with torch.no_grad():
    for batch in tqdm(test_loader):

      input_ids = batch["input_ids"].to(device)
      attention_mask = batch["attention_mask"].to(device)
      token_type_ids = batch["token_type_ids"].to(device)
      true_labels =  batch["labels"].to(device)




      outputs = model.forward(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids
      )

      logits = outputs[0]
    
      _, true_labels_idx = torch.max(true_labels, dim = 1)
      _, pred_labels_idx = torch.max(logits, dim = 1)
      info_logger.info(pred_labels_idx)
      info_logger.info(true_labels_idx)


      metrics(preds=pred_labels_idx, target=true_labels_idx)

  metrics(preds=torch.tensor([0, 1]).to(device), target=torch.tensor([1, 0]).to(device))
  print(metrics.compute())