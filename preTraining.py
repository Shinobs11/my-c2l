import json, os, pickle, torch, logging, typing, numpy as np, glob, argparse
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from tqdm import tqdm
from transformers import BertTokenizerFast, BertForSequenceClassification, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup

from src.utils.pickleUtils import pdump, pload
from src.proecssing import correct_count
from transformers.models.bert.modeling_bert import SequenceClassifierOutput
from src.classes.model import BertForCounterfactualRobustness
from  src.classes.datasets import ClassificationDataset
import pickle


import json, psutil

from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy
import wandb
torch.manual_seed(0)




def pretrainBERT(
  dataset_name: str,
  batch_size: int,
  epoch_num: int,
  use_pinned_memory: bool
):


  DATASET_NAME = dataset_name
  BATCH_SIZE = batch_size
  EPOCH_NUM = epoch_num


  DATASET_PATH = f"datasets/{DATASET_NAME}/base"
  OUTPUT_PATH = f"checkpoints/{DATASET_NAME}/model"






# memAvailable = psutil.virtual_memory().available
# estimatedMemConsumed = os.path.getsize(os.path.join(DATASET_PATH, "train_set.pickle.blosc")) * 3
# USE_PINNED_MEMORY = True if (args.use_pinned_memory & (memAvailable > estimatedMemConsumed)) == 1 else False




  def loadTrainData():


    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    train_set = pload(os.path.join(DATASET_PATH, "train_set"))[0:10]
    train_labels:list = train_set['label'].tolist()
    train_texts:list[str] = train_set['text'].tolist()
    

    
    train_encodings = tokenizer(train_texts, padding=True, truncation=True)
    pdump(train_encodings, os.path.join(DATASET_PATH, "train_encodings"))
    
    train_dataset = ClassificationDataset(labels=train_labels, encodings=train_encodings)
    train_loader = DataLoader(
      train_dataset,
      batch_size=BATCH_SIZE,
      shuffle=False, #TODOS: test impact of shuffle
      pin_memory=use_pinned_memory
      )
    return train_loader

  def loadValData():
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    valid_set = pload(os.path.join(DATASET_PATH, "valid_set"))[0:10]
    valid_texts:list[str] = valid_set['text'].tolist()
    valid_labels: list = valid_set['label'].tolist()
    valid_encodings = tokenizer(valid_texts, padding=True, truncation=True)
    pdump(valid_encodings, os.path.join(DATASET_PATH, "valid_encodings"))

    valid_dataset = ClassificationDataset(labels=valid_labels, encodings=valid_encodings)
    valid_loader = DataLoader(
      valid_dataset,
      batch_size=BATCH_SIZE,
      shuffle=False,
      pin_memory=use_pinned_memory
    )

    return valid_loader



  def pretrainModel():

    

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if os.path.exists(os.path.join(DATASET_PATH, "trainLoader.pickle.blosc")):
      train_loader: DataLoader = pload(os.path.join(DATASET_PATH, "trainLoader"))
    else:
      train_loader: DataLoader = loadTrainData()
    if os.path.exists(os.path.join(DATASET_PATH, "validLoader.pickle.blosc")):
      valid_loader: DataLoader = pload(os.path.join(DATASET_PATH, "validLoader"))
    else:
      valid_loader: DataLoader = loadValData()
    

    num_classes = -1
    if len(train_loader.dataset[0]["label"].shape) == 1:
      num_classes = train_loader.dataset[0]["label"].shape[0]
    else:
      s = set()
      [s.add(x["labels"]) for x in train_loader.dataset]
      num_classes = len(s)



    torch.cuda.empty_cache()
    model:torch.nn.Module = BertForCounterfactualRobustness.from_pretrained('bert-base-uncased', num_labels=num_classes)  # type: ignore
    # model = torch.compile(model) #type: ignore
    # model:torch.nn.Module = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_classes) #type: ignore  
    # model:torch.nn.Module = torch.compile(model) #type: ignore
    # model:BertForSequenceClassification = torch.nn.DataParallel(model)  # type: ignore # ! only use with distributed computing
    model.to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=5e-5)
    # scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=50, num_training_steps=len(train_loader)*EPOCH_NUM)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=3, verbose=True)
    warmup = get_constant_schedule_with_warmup(optim, num_warmup_steps=int(len(train_loader)*0.10))
    steps = 0
    best_acc = 0
    best_epoch = -1
    all_loss = []



    for epoch in range(EPOCH_NUM):
      train_metrics: MulticlassAccuracy  = MulticlassAccuracy(num_classes=num_classes).to(device)
      valid_metrics: MulticlassAccuracy = MulticlassAccuracy(num_classes=num_classes).to(device)
      model.train() # switch to training mode
      train_progress_bar = tqdm(train_loader)

      for batch in train_progress_bar:
        optim.zero_grad()
        
        input_ids = batch['input_ids'].to(device, dtype=torch.long)
        attention_mask = batch['attention_mask'].to(device, dtype=torch.long)
        token_type_ids = batch['token_type_ids'].to(device, dtype=torch.long)
        labels = batch['label'].to(device)
        true_labels = labels #shape = [n_labels]
        _, labels = torch.max(labels, dim = 1)  


        outputs:SequenceClassifierOutput|tuple[torch.Tensor] = model.forward(
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


        loss.sum().backward()
        # epoch_loss.append(loss.sum())
        train_progress_bar.set_description("Epoch train_accuracy %f" % train_metrics.compute().item())
        optim.step()
        warmup.step()
        # scheduler.step()
        steps+=1

      # all_loss.append(epoch_loss)
      model.eval()# setting model to evaluation mode



      accuracy = 0.0
      val_loss = 0.0
      for batch in valid_loader:
        with torch.no_grad():
          # labels = batch['label'].to(device)
          input_ids = batch['input_ids'].to(device, dtype=torch.long)
          attention_mask = batch['attention_mask'].to(device, dtype=torch.long)
          token_type_ids = batch['token_type_ids'].to(device, dtype=torch.long)
          true_labels = batch['label'].to(device)
          outputs:SequenceClassifierOutput|tuple[torch.Tensor] = model.forward(
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
          # print("Preds:",pred_labels_idx)
          # print("Trues:",true_labels_idx)
          valid_metrics.update(pred_labels_idx, true_labels_idx)
  
        accuracy = valid_metrics.compute().item()
      if accuracy >= best_acc:
        best_epoch = epoch
        best_acc = accuracy
      print(f"Accuracy: {accuracy}")
      scheduler.step(val_loss)
      wandb.log({"accuracy": accuracy, "valid_loss": val_loss})
      model.save_pretrained(os.path.join(OUTPUT_PATH, f"epoch_{epoch}")) #type:ignore
    pdump(all_loss, os.path.join(OUTPUT_PATH,"training_loss"))
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



  pretrainModel()




### Configuring misc. stuff
from transformers import logging
logging.set_verbosity_error()
