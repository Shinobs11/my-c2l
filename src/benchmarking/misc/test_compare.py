from itertools import product
import torch, timeit
import torch.utils.benchmark as benchmark

def batched_dot_mul_sum(a, b):
  return a.mul(b).sum(-1)

def batched_dot_bmm(a, b):
  a = a.reshape(-1, 1, a.shape[-1])
  b =  b.reshape(-1, b.shape[-1], 1)
  return torch.bmm(a, b).flatten(-3)

x = torch.randn(10000, 64)

assert batched_dot_mul_sum(x, x).allclose(batched_dot_bmm(x, x))

# Compare takes a list of measurements which we'll save in results.
results = []

# sizes = [1, 64, 1024]
# for b, n in product(sizes, sizes):
#     print(f"Batch {b} {n}")
#     # label and sub_label are the rows
#     # description is the column
#     label = 'Batched dot'
#     sub_label = f'[{b}, {n}]'
#     x = torch.ones((b, n))
#     for num_threads in [1, 4, 16]:
#         results.append(benchmark.Timer(
#             stmt='batched_dot_mul_sum(x, x)',
#             setup='from __main__ import batched_dot_mul_sum',
#             globals={'x': x},
#             num_threads=num_threads,
#             label=label,
#             sub_label=sub_label,
#             description='mul/sum',
#         ).blocked_autorange(min_run_time=0.2))
#         results.append(benchmark.Timer(
#             stmt='batched_dot_bmm(x, x)',
#             setup='from __main__ import batched_dot_bmm',
#             globals={'x': x},
#             num_threads=num_threads,
#             label=label,
#             sub_label=sub_label,
#             description='bmm',
#         ).blocked_autorange(min_run_time=0.2))

# compare = benchmark.Compare(results)
# compare.trim_significant_figures()
# compare.colorize()
# compare.print()

import pickle

ab_test_results = []
for env in ('environment A: mul/sum', 'environment B: bmm'):
    for b, n in ((1, 1), (1024, 10000), (10000, 1)):
        x = torch.ones((b, n))
        dot_fn = (batched_dot_mul_sum if env == 'environment A: mul/sum' else batched_dot_bmm)
        m = benchmark.Timer(
            stmt='batched_dot(x, x)',
            globals={'x': x, 'batched_dot': dot_fn},
            num_threads=1,
            label='Batched dot',
            description=f'[{b}, {n}]',
            env=env,
        ).blocked_autorange(min_run_time=0.2)
        ab_test_results.append(pickle.dumps(m))

ab_results = [pickle.loads(i) for i in ab_test_results]
compare = benchmark.Compare(ab_results)
compare.trim_significant_figures()
compare.colorize()
compare.print()