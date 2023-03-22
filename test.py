import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch
import os


def test_fn(index):
  return



if __name__ == "__main__":
  xmp.spawn(test_fn)