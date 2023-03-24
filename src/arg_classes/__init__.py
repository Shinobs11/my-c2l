from torch.utils.data import DataLoader, Dataset


class Args_PT():
  dataset_name: str
  batch_size: int
  epoch_num: int
  use_pinned_memory: bool
  use_wandb: bool
  num_workers:int

  def __init__(self, dataset_name, batch_size, epoch_num, use_pinned_memory, use_wandb, num_workers = 4):
    self.dataset_name = dataset_name
    self.batch_size = batch_size
    self.epoch_num = epoch_num
    self.use_pinned_memory = use_pinned_memory
    self.use_wandb = use_wandb
    self.num_workers = num_workers


class Args_Map_PT():

  train_dataset: Dataset
  valid_dataset: Dataset
  num_classes: int


  def __init__(self, args_pt: Args_PT, num_clases: int, train_dataset: Dataset, valid_dataset: Dataset):
    self.args_pt = args_pt
    self.num_classes = num_clases
    self.train_dataset = train_dataset
    self.valid_dataset = valid_dataset


