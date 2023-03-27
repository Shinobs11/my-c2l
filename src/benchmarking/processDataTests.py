# import torch
# from torch import Tensor
# import inspect
# import os
# import functorch

# cwd = os.path.split(__file__)[0]



# def filter_sep_tokens_setup():
#   from torch.utils.data import DataLoader
#   from .testData import tokenizer, train_dataset
#   train_loader = DataLoader(
#     dataset=train_dataset,
#     batch_size=20,
#     shuffle=False,
#     pin_memory=False,
#     persistent_workers=False
#   )

#   def test():
#     for batch in train_loader:
#       tokens = torch.tensor(
#         [x for x in batch['input_ids'][0][1:] if x not in [tokenizer.sep_token_id, tokenizer.pad_token_id]]
#         )


#   return inspect.currentframe()




# def mask_candidate_selection_setup():
#   from ..utils.pickleUtils import pload

#   importance_indices, topk_logit_indices = pload(os.path.join(cwd, "test_data", "generate_triplets_299"))
#   importance_indices: Tensor = importance_indices
#   topk_logit_indices: Tensor = topk_logit_indices

#   def test():      
#     mask_candidates = [topk_logits[importance_index + 1] for importance_index, topk_logits in zip(importance_indices, topk_logit_indices)]
#     return mask_candidates
#   print(test())
#   return inspect.currentframe()
# mask_candidate_selection_setup()

# def new_mask_candidate_selection_setup():
#   from ..utils.pickleUtils import pload
#   def func_mc(mc:Tensor, idx: Tensor):
#     print(idx.shape)
#     return mc.index_select(dim=0, index=idx).squeeze(0)
  
#   batch_func = functorch.vmap(func_mc, in_dims=0, out_dims=0)
#   importance_indices, topk_logit_indices = pload(os.path.join(cwd, "test_data", "generate_triplets_299"))
#   importance_indices: Tensor = importance_indices
#   topk_logit_indices: Tensor = topk_logit_indices
  
  

#   def test():
#     importance_idx = importance_indices.add(1).unsqueeze(1)
#     print(importance_idx)
#     mask_candidates = batch_func(topk_logit_indices, importance_idx)
#     return mask_candidates
#   print(test())
  
#   return inspect.currentframe()
# new_mask_candidate_selection_setup()