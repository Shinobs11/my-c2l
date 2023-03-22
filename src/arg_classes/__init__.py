class Args_PT():
  dataset_name: str
  batch_size: int
  epoch_num: int
  use_pinned_memory: bool
  use_wandb: bool

  def __init__(self, dataset_name, batch_size, epoch_num, use_pinned_memory, use_wandb):
    self.dataset_name = dataset_name
    self.batch_size = batch_size
    self.epoch_num = epoch_num
    self.use_pinned_memory = use_pinned_memory
    self.use_wandb = use_wandb
    
