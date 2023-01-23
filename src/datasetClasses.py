import torch
from torch import Tensor
from torch.utils.data import Dataset
from typing import TypedDict

class DataItem(TypedDict):
  input_ids: Tensor
  attention_mask: Tensor
  token_type_ids: Tensor
  labels: Tensor

# DataItem = TypedDict('DataItem', {
#   'input_ids':Tensor,
#   'attention_mask': Tensor,
#   'input_type_ids': Tensor,
#   'labels': Tensor
# },
# total=True
# )

class IMDBDataset(Dataset):
  def __init__(self, encodings, labels):
    self.labels = labels
    self.encodings = encodings
  
  def __getitem__(self, idx):
    
    item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()} #
    # item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
    item['labels'] = torch.tensor(self.labels[idx])
    return item


  def __len__(self):
    return len(self.labels)


