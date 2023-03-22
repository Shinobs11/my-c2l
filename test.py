import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch
import os


def test_fn(index):
  print("Hello world from process {}".format(index))


if __name__ == '__main__':

  xmp.spawn(test_fn, args=(), start_method='spawn', join=True)