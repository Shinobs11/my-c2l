import os, torch
from transformers import BertTokenizerFast, get_constant_schedule_with_warmup
from src.arg_classes import Args_Map_PT, Args_PT
from src.utils.pickleUtils import pdump, pload
from transformers.models.bert.modeling_bert import SequenceClassifierOutput 
from src.classes.model import BertForCounterfactualRobustness
from src.classes.datasets import ClassificationDataset
from torch.utils.data import DataLoader

import torch.distributed as dist
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl
from torch.distributed.optim.zero_redundancy_optimizer import ZeroRedundancyOptimizer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from typing import Tuple, List, Union


import torch_xla.experimental.pjrt_backend
import torch_xla.experimental.pjrt as pjrt



def loadTrainData(dataset_path: str):
  tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
  train_set = pload(os.path.join(dataset_path, "train_set"))[0:256]
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


def map_pt(index, args_map: Args_Map_PT):
  args = args_map.args_pt
  dist.init_process_group('xla', init_method='pjrt://')
  # dist.init_process_group('xla', world_size=xm.xrt_world_size(), rank=xm.get_ordinal())
  device = xm.xla_device()
  print(f"Started process on index {index}")



  DATASET_NAME = args.dataset_name
  BATCH_SIZE = args.batch_size
  EPOCH_NUM = args.epoch_num
  USE_PINNED_MEMORY = args.use_pinned_memory
  USE_WANDB = args.use_wandb
  NUM_WORKERS= args.num_workers
  NUM_CLASSES = args_map.num_classes
  DATASET_PATH = f"datasets/{DATASET_NAME}/base"
  OUTPUT_PATH = f"checkpoints/{DATASET_NAME}/model"
  LOG_MODULUS = args.log_modulus

  #acquire tpu device
  
  #initialize distributed package with url


  #stops all child workers here and waits for master to rendezvous
  # if not xm.is_master_ordinal():
  #   xm.rendezvous('dataset_prep')
  # print(f"Index {index} has entered the rendezvous dataset_prep")
  

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
  # print(f"Index {index} has created samplers")


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


  # print(f"Index {index} has created parallel loaders")
  #master rendezvous here and allows execution to continue.
  # if xm.is_master_ordinal():
  #   xm.rendezvous('dataset_prep')




  model:torch.nn.Module = BertForCounterfactualRobustness.from_pretrained('bert-base-uncased', num_labels=NUM_CLASSES).to(device)  # type: ignore
  # pjrt.broadcast_master_param(model)
  model = DDP(model, gradient_as_bucket_view=True, broadcast_buffers=False) 
  # if xm.get_ordinal() == 0:
    # pjrt.broadcast_master_param(model)
  #   xm.rendezvous('broadcast_parameters')
  # else:
  #   xm.rendezvous('broadcast_parameters')
  


  #TODOS: Figure out if DDP is useful here



  
  optim = ZeroRedundancyOptimizer(
    params=model.parameters(),
    optimizer_class=torch.optim.AdamW,
    # parameters_as_bucket_view=True,
    lr=5e-5
  )
  # scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=50, num_training_steps=len(train_loader)*EPOCH_NUM)
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=3, verbose=True)
  warmup = get_constant_schedule_with_warmup(optim, num_warmup_steps=int(len(train_loader)*0.10))
  best_acc = 0
  best_epoch = -1

  for epoch in range(EPOCH_NUM):
    # train_metrics: MulticlassAccuracy  = MulticlassAccuracy(num_classes=args_map.num_classes).to(device)
    # valid_metrics: MulticlassAccuracy = MulticlassAccuracy(num_classes=args_map.num_classes).to(device)
    model.train() # switch to training mode
    # train_progress_bar = tqdm(train_loader)
    # tracker = xm.RateTracker()
    for step, batch in enumerate(train_loader):
      
      optim.zero_grad()
      
      input_ids = batch['input_ids'].to(device, dtype=torch.long)
      attention_mask = batch['attention_mask'].to(device, dtype=torch.long)
      token_type_ids = batch['token_type_ids'].to(device, dtype=torch.long)
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

      print(f"step: {step}")


      # train_metrics.update(pred_labels_idx, true_labels_idx)

      
      loss.sum().backward()

      #* Somehwere in here is the AllReduce issue *
      
      xm.optimizer_step(optim, pin_layout=False)
      #* End here*
      warmup.step()

def pretrainBERT(
  args: Args_PT,
):
  # os.environ['XRT_TPU_CONFIG'] = 'localservice;0;localhost:51011'
  # os.environ['MASTER_ADDR'] = 'localhost'
  # os.environ['MASTER_PORT'] = '12355'
  os.environ['PJRT_DEVICE'] = 'TPU'
  os.environ['XLA_USE_BF16'] = "1"

  print(f"My name is {__name__}")


  if xm.pjrt.using_pjrt():
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


  if(__name__ == "preTrainingv2"):
    xmp.spawn(map_pt, args=(args_map,), start_method='spawn', join=True)


