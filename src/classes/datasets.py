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
    # self.special_tokens = special_tokens
    # self.special_token_ids = special_token_ids
  
    # self.token_counts = []
    # for encoding in encodings['input_ids']:
    #   try:
    #     token_count = encoding.index(0) + 1
    #   except:
    #     token_count = len(encoding)
    #     pass
    #   self.token_counts.append(token_count)
    

    # self.special_token_pos_list = [
    #   [np.where(token_id == np.array(encoding))
    #     for token_id in
    #     self.special_token_ids ] for encoding in encodings['input_ids']
    # ]

    # self.special_token_dict  = {
    #   'unk_token_pos': [self.special_token_pos_list[idx][0] for idx in range(len(self.special_token_pos_list))],
    #   'sep_token_pos': [self.special_token_pos_list[idx][1] for idx in range(len(self.special_token_pos_list))],
    #   'pad_token_pos': [self.special_token_pos_list[idx][2] for idx in range(len(self.special_token_pos_list))],
    #   'cls_token_pos': [self.special_token_pos_list[idx][3] for idx in range(len(self.special_token_pos_list))],
    #   'mask_token_pos': [self.special_token_pos_list[idx][4] for idx in range(len(self.special_token_pos_list))]
    # }

  # ['[UNK]', '[SEP]', '[PAD]', '[CLS]', '[MASK]']
  # {'unk_token': '[UNK]', 'sep_token': '[SEP]', 'pad_token': '[PAD]', 'cls_token': '[CLS]', 'mask_token': '[MASK]'}
  def __getitem__(self, idx):
    item = {
      'labels': torch.tensor(self.labels[idx]),
      # 'token_count': torch.tensor(self.token_counts[idx]),
    }
    item.update({k: torch.tensor(v[idx]) for k, v in self.encodings.items()})



    return item


  def __len__(self):
    return len(self.labels)


# class MaskedIMDBDataset(Dataset):
#   def __init__(self, unmasked_dataset: Dataset, top_k: int):
#     self.unmasked_dataset = unmasked_dataset
#     self.top_k = top_k

#   def __getitem__(self, idx):
#     item = self.unmasked_dataset.__getitem__(idx)
#     encodings_keys = {'input_ids', 'attention_mask', 'token_type_ids'} 
#     ret_item = dict()
#     for k, v in item.items():
#       if k in encodings_keys:
#         ret_item.update({
#           k: v.expand(self.top_k, *v.shape)
#         })
#       else:
#         ret_item.update({k: v})
#     return ret_item





class CFIMDbDataset(Dataset):
    def __init__(self, anchor_encodings, positive_encodings, negative_encodings, labels):
        self.anchor_encodings = anchor_encodings
        self.positive_encodings = positive_encodings
        self.negative_encodings = negative_encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = dict()
        item.update({'anchor_'+key: torch.tensor(val[idx]) for key, val in self.anchor_encodings.items()})
        item.update({'positive_'+key: torch.tensor(val[idx]) for key, val in self.positive_encodings.items()})
        item.update({'negative_'+key: torch.tensor(val[idx]) for key, val in self.negative_encodings.items()})
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
        
#* From here down I decided to just copy and paste it because I wasn't sure what some of it was for.
class CFClassifcationDataset(Dataset):
    def __init__(self, anchor_encodings, positive_encodings, negative_encodings, triplet_sample_masks, labels):
        self.anchor_encodings = anchor_encodings
        self.positive_encodings = positive_encodings
        self.negative_encodings = negative_encodings
        self.triplet_sample_masks = triplet_sample_masks
        self.labels = labels

    def __getitem__(self, idx):
        item = dict()
        item.update({'anchor_'+key: torch.tensor(val[idx]) for key, val in self.anchor_encodings.items()})
        item.update({'positive_'+key: torch.tensor(val[idx]) for key, val in self.positive_encodings.items()})
        item.update({'negative_'+key: torch.tensor(val[idx]) for key, val in self.negative_encodings.items()})
        item['triplet_sample_masks'] = torch.tensor(self.triplet_sample_masks[idx])
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

