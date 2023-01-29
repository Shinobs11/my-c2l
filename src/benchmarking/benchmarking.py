import torch.utils.benchmark as bench
import inspect
from .processDataTests import filter_sep_tokens_setup

print(filter_sep_tokens_setup)

# def test_function():
#   ctx = filter_sep_tokens_setup()
#   assert ctx != None

#   t0 = bench.Timer(
#     stmt=inspect.getsource(filter_sep_tokens_setup.__name__),
#     setup='from __main__ import filter_sep_tokens_setup',
#     globals={
#       **(ctx.f_locals)
#     }
#   )
#   print(t0.timeit(5))

# test_function()