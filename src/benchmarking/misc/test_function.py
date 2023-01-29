import torch






def batched_dot_mul_sum(a, b):
  return a.mul(b).sum(-1)

def batched_dot_bmm(a, b):
  a = a.reshape(-1, 1, a.shape[-1])
  b =  b.reshape(-1, b.shape[-1], 1)
  return torch.bmm(a, b).flatten(-3)



# x = torch.randn(10000, 64)

# assert batched_dot_mul_sum(x, x).allclose(batched_dot_bmm(x, x))
