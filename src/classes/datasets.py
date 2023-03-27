from re import M
import torch
from torch import Tensor
from torch.utils.data import Dataset
from typing import TypedDict
import numpy as np
class DataItem(TypedDict):
  input_ids: Tensor
  attention_mask: Tensor
  token_type_ids: Tensor
  labels: Tensor


class ClassificationDataset(Dataset):
  def __init__(self, encodings, labels, enumerate=False):
    self.labels = labels
    self.encodings = encodings
    self.enumerate = enumerate
  def __getitem__(self, idx):
    item = {
      'label': torch.tensor(self.labels[idx]),
    }
    item.update({k: torch.tensor(v[idx]) for k, v in self.encodings.items()})


    if (self.enumerate == True):
      return idx, item
    else:
       return item

  def __len__(self):
    return len(self.labels)


class TripletGenDataset(Dataset):
  def __init__(self, encodings, importance, enumerate=False):
    self.encodings = encodings
    self.importance = importance
    self.enumerate = enumerate
  def __getitem__(self, idx):
    item = {
      'importance_indices': self.importance[idx].clone().detach(),
    }


    item.update({k: v[idx].clone().detach() for k, v in self.encodings.items()})

    if (self.enumerate == True):
      return idx, item
    else:
       return item

  def __len__(self):
    return len(self.importance)



class CFClassifcationDataset(Dataset):
    def __init__(self, anchor_encodings, positive_encodings, negative_encodings, triplet_sample_masks, labels, enumerate=False):
        self.anchor_encodings = anchor_encodings
        self.positive_encodings = positive_encodings
        self.negative_encodings = negative_encodings
        self.triplet_sample_masks = triplet_sample_masks
        self.labels = labels
        self.enumerate = enumerate

    def __getitem__(self, idx):
        item = dict()
        item.update({'anchor_'+key: torch.tensor(val[idx]) for key, val in self.anchor_encodings.items()})
        item.update({'positive_'+key: torch.tensor(val[idx]) for key, val in self.positive_encodings.items()})
        item.update({'negative_'+key: torch.tensor(val[idx]) for key, val in self.negative_encodings.items()})
        item['triplet_sample_mask'] = torch.tensor(self.triplet_sample_masks[idx])
        item['label'] = torch.tensor(self.labels[idx])
        if (self.enumerate == True):
          return idx, item
        else:
          return item

    def __len__(self):
        return len(self.labels)

