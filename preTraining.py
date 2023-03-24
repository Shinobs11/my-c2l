import json, os, pickle, torch, logging, typing, numpy as np, glob, argparse
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from tqdm import tqdm
from transformers import BertTokenizerFast, BertForSequenceClassification, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
from src.arg_classes import Args_Map_PT, Args_PT

from src.utils.pickleUtils import pdump, pload
from src.proecssing import correct_count
from transformers.models.bert.modeling_bert import SequenceClassifierOutput
from src.classes.model import BertForCounterfactualRobustness
from src.classes.datasets import ClassificationDataset
from src.logging.logSetup import logSetup
from src.arg_classes import Args_PT


import pickle

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl


from typing import Tuple, List, Union
import json, psutil

from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy
import wandb
import torch_xla.experimental.pjrt_backend
import torch_xla.experimental.pjrt as pjrt
torch.manual_seed(0)






def loadTrainData(dataset_path: str):
  tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
  train_set = pload(os.path.join(dataset_path, "train_set"))
  train_labels:list = train_set['label'].tolist()
  train_texts:list[str] = train_set['text'].tolist()
  

  
  train_encodings = tokenizer(train_texts, padding=True, truncation=True)
  pdump(train_encodings, os.path.join(dataset_path, "train_encodings"))
  
  train_dataset = ClassificationDataset(labels=train_labels, encodings=train_encodings)

  return train_dataset

def loadValData(dataset_path: str):
  tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

  valid_set = pload(os.path.join(dataset_path, "valid_set"))
  valid_texts:list[str] = valid_set['text'].tolist()
  valid_labels: list = valid_set['label'].tolist()
  valid_encodings = tokenizer(valid_texts, padding=True, truncation=True)
  pdump(valid_encodings, os.path.join(dataset_path, "valid_encodings"))

  valid_dataset = ClassificationDataset(labels=valid_labels, encodings=valid_encodings)


  return valid_dataset



#TODOS: Rewrite as class or non-nested function, issue might be that function is pickled and transferred elsewhere
#But if the function can't be pickled, then the function can't be transferred.




def map_pt(index, args_map: Args_Map_PT):
  args = args_map.args_pt
  # dist.init_process_group('xla', init_method='pjrt://')
  dist.init_process_group('xla', world_size=xm.xrt_world_size(), rank=xm.get_ordinal())
  device = xm.xla_device()
  print(f"Started process on index {index}")
  log = logSetup(f"logs/pt/{args.dataset_name}_{index}", f"logs/pt/{args.dataset_name}_{index}.log", logging.INFO, None)
  log.info(f"Started process on index {index}")

  log.info(f"Device: {device}")
  
  test_tensor = torch.tensor([1,2,3,4,5]).to(device)
  test_tensor = test_tensor.sum()
  log.info(f"Test tensor post-sum: {test_tensor}")
  res_tensor = xm.all_reduce(xm.REDUCE_SUM, [test_tensor], scale=1.0)
  log.info(f"Result tensor: {res_tensor}")





  def pretrainModel():

    DATASET_NAME = args.dataset_name
    BATCH_SIZE = args.batch_size
    EPOCH_NUM = args.epoch_num
    USE_PINNED_MEMORY = args.use_pinned_memory
    USE_WANDB = args.use_wandb
    NUM_WORKERS= args.num_workers
    NUM_CLASSES = args_map.num_classes
    DATASET_PATH = f"datasets/{DATASET_NAME}/base"
    OUTPUT_PATH = f"checkpoints/{DATASET_NAME}/model"


    #acquire tpu device
   
    #initialize distributed package with url


    #stops all child workers here and waits for master to rendezvous
    # if not xm.is_master_ordinal():
    #   xm.rendezvous('dataset_prep')
    print(f"Index {index} has entered the rendezvous dataset_prep")
   

    if xm.xrt_world_size() > 1:
      train_sampler = DistributedSampler(
          args_map.train_dataset,
          num_replicas=xm.xrt_world_size(),
          rank=xm.get_ordinal(),
          shuffle=False
        )
      valid_sampler = DistributedSampler(
          args_map.valid_dataset,
          num_replicas=xm.xrt_world_size(),
          rank=xm.get_ordinal(),
          shuffle=False
        )
    else:
      train_sampler = None
      valid_sampler = None
    print(f"Index {index} has created samplers")


    train_loader = DataLoader(
    args_map.train_dataset, 
    batch_size=BATCH_SIZE, 

    num_workers=NUM_WORKERS, 
    pin_memory=USE_PINNED_MEMORY,
    sampler=train_sampler
    )
    valid_loader = DataLoader(
    args_map.valid_dataset,
    batch_size=BATCH_SIZE, 
    num_workers=NUM_WORKERS, 
    pin_memory=USE_PINNED_MEMORY,
    sampler=valid_sampler
    )
    print(f"Index {index} has created data loaders")


    train_loader = pl.MpDeviceLoader(train_loader, device)
    valid_loader = pl.MpDeviceLoader(valid_loader, device)


    print(f"Index {index} has created parallel loaders")
    #master rendezvous here and allows execution to continue.
    # if xm.is_master_ordinal():
    #   xm.rendezvous('dataset_prep')




    model:torch.nn.Module = BertForCounterfactualRobustness.from_pretrained('bert-base-uncased', num_labels=NUM_CLASSES).to(device)  # type: ignore
    # pjrt.broadcast_master_param(model)
    model = DDP(model) 
    # if xm.get_ordinal() == 0:
      # pjrt.broadcast_master_param(model)
    #   xm.rendezvous('broadcast_parameters')
    # else:
    #   xm.rendezvous('broadcast_parameters')
    


    #TODOS: Figure out if DDP is useful here


    optim = torch.optim.AdamW(model.parameters(), lr=5e-5)
    # scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=50, num_training_steps=len(train_loader)*EPOCH_NUM)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=3, verbose=True)
    warmup = get_constant_schedule_with_warmup(optim, num_warmup_steps=int(len(train_loader)*0.10))
    steps = 0
    best_acc = 0
    best_epoch = -1
    all_loss = []


    return
    for epoch in range(EPOCH_NUM):
      train_metrics: MulticlassAccuracy  = MulticlassAccuracy(num_classes=args_map.num_classes).to(device)
      valid_metrics: MulticlassAccuracy = MulticlassAccuracy(num_classes=args_map.num_classes).to(device)
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


        outputs:Union[SequenceClassifierOutput, Tuple[torch.Tensor]] = model(
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


        xm.optimizer_step(optim)

        warmup.step()
        xm.mark_step()
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
          outputs:Union[SequenceClassifierOutput, Tuple[torch.Tensor]] = model(
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


      xm.master_print(f"Accuracy: {accuracy}")
      scheduler.step(val_loss)

      if xm.is_master_ordinal():
        if(USE_WANDB):
          wandb.log({"valid accuracy": accuracy, "valid loss": val_loss})
        model.save_pretrained(os.path.join(OUTPUT_PATH, f"epoch_{epoch}")) #type:ignore
    # pdump(all_loss, os.path.join(OUTPUT_PATH,"training_loss"))
    xm.master_print(f"\nBest model is epoch {best_epoch}.")
    xm.master_print(f"\nBest accuracy is {best_acc}")

    if xm.is_master_ordinal():
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





  
def pretrainBERT(
  args: Args_PT,
):
  os.environ['XRT_TPU_CONFIG'] = 'localservice;0;localhost:51011'
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = '12355'
  # os.environ['PJRT_DEVICE'] = 'TPU'

  print(f"My name is {__name__}")


  if pjrt.using_pjrt():
    print("pjrt: enabled")
  else:
    print("pjrt: disabled")


  DATASET_NAME = args.dataset_name
  BATCH_SIZE = args.batch_size
  EPOCH_NUM = args.epoch_num
  USE_PINNED_MEMORY = args.use_pinned_memory
  NUM_WORKERS = args.num_workers
  USE_WANDB = args.use_wandb

  DATASET_PATH = f"datasets/{DATASET_NAME}/base"
  OUTPUT_PATH = f"checkpoints/{DATASET_NAME}/model"

  if not os.path.exists(f"logs/pt"):
    os.makedirs(f"logs/pt")




  train_dataset  = loadTrainData(DATASET_PATH)
  valid_dataset = loadValData(DATASET_PATH)

  num_classes = -1
  if len(train_dataset[0]["label"].shape) == 1: #type:ignore
    num_classes = train_dataset[0]["label"].shape[0] #type:ignore
  else:
    s = set()
    [s.add(x["labels"]) for x in train_dataset] #type:ignore
    num_classes = len(s)


  args_map = Args_Map_PT(
    args_pt=args,
    train_dataset=train_dataset,
    valid_dataset=valid_dataset,
    num_clases=num_classes
  )


  if(__name__ == "preTraining"):
    xmp.spawn(map_pt, args=(args_map,), start_method='fork', join=True)





# ### Configuring misc. stuff
# from transformers import logging
# logging.set_verbosity_error()
