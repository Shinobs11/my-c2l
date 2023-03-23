import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch
import os
import torch_xla.core.functions as xmfunc
import torch.distributed as dist
import torch
import os
from src.logging.logSetup import logSetup
from src.arg_classes import Args_PT
import logging

import torch_xla.experimental.pjrt_backend
import torch_xla.experimental.pjrt as pjrt


def test_fn(index, args):
  dist.init_process_group('xla', init_method='pjrt://')




  xm.master_print(f"MasterPrint: Started process on index {index}")
  xm.master_print(f"My name is {__name__}")
  print(f"Started process on index {index}")
  log = logSetup(f"logs/pt/{args.dataset_name}_{index}", f"logs/pt/{args.dataset_name}_{index}.log", logging.INFO, None)
  log.info(f"Started process on index {index}")
  device = xm.xla_device()
  log.info(f"Device: {device}")
  
  test_tensor = torch.tensor([1,2,3,4,5]).to(device)
  test_tensor = test_tensor.sum()
  log.info(f"Test tensor post-sum: {test_tensor}")




  res_tensor = xmfunc.all_reduce(xm.REDUCE_SUM, [test_tensor], scale=1.0)
  log.info(f"Result tensor: {res_tensor}")


if __name__ == '__main__':
  os.environ['XRT_TPU_CONFIG'] = 'localservice;0;localhost:51011'
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = '12355'
  os.environ['PJRT_DEVICE'] = 'TPU'

  
  print(f"My name is {__name__}")
  # xmp.spawn(main, nprocs=1, start_method='spawn')
  args_pt = Args_PT(
    dataset_name="yelp_nyc",
    batch_size=8,
    epoch_num=20,
    use_pinned_memory=False,
    use_wandb=False
  )

  xmp.spawn(test_fn, args=(args_pt,), start_method='spawn', join=True)

