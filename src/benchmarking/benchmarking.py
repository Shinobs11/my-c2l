# import torch.utils.benchmark as bench
# import inspect
# from .processDataTests import filter_sep_tokens_setup
# import typing
# import types
# import os
# from ..utils.pickleUtils import pload
# print(filter_sep_tokens_setup)
# cwd = os.path.split(__file__)[0]
# # def test_function():
  
# #   ctx = filter_sep_tokens_setup()
# #   assert ctx != None
# #   importance_indices, topk_logit_indices = pload(os.path.join(cwd, "test_data", "generate_triplets_299"))
# #   t0 = bench.Timer(
# #     stmt="mask_candidates = [topk_logits[importance_index + 1] for importance_index, topk_logits in zip(importance_indices, topk_logit_indices)]",
# #     globals={
# #       "importance_indices": importance_indices,
# #       "topk_logit_indices": topk_logit_indices
# #     }
# #   )
# #   # print(t0.timeit(5))
# #   print(t0.blocked_autorange())
# #   # print(ctx.f_locals)

# # test_function()

# def simple_benchmark(ctx_func: typing.Callable[[], types.FrameType|None]):
#     ctx = ctx_func()
#     assert ctx != None
#     t0 = bench.Timer(
#         stmt="test()",
#         globals = {**(ctx.f_locals)}
#     )
#     # print(t0.blocked_autorange())
#     print(t0.timeit(100000))


# from .processDataTests import mask_candidate_selection_setup, new_mask_candidate_selection_setup
# simple_benchmark(mask_candidate_selection_setup)
# simple_benchmark(new_mask_candidate_selection_setup)