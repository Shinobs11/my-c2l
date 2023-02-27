import torch, numpy as np
from torch import Tensor
from .classes.datasets import DataItem
from torch.utils.data import DataLoader
from tqdm import tqdm
def correct_count(logits:Tensor, labels:Tensor):
  _, indices = torch.max(logits, dim=1)
  _, label_indices = torch.max(labels, dim=1)
  correct = torch.sum(indices == label_indices)
  return correct.item()








