import torch
import inspect



def filter_sep_tokens_setup():
  from torch.utils.data import DataLoader
  from .testData import tokenizer, train_dataset
  train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=20,
    shuffle=False,
    pin_memory=False,
    persistent_workers=False
  )

  def filter_sep_tokens_test():
    for batch in train_loader:
      tokens = torch.tensor(
        [x for x in batch['input_ids'][0][1:] if x not in [tokenizer.sep_token_id, tokenizer.pad_token_id]]
        )


  return inspect.currentframe()