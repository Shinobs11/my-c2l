from datetime import datetime
import multiprocessing
import os
import sys
import time
import unittest

import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
import torch_xla.debug.metrics_compare_utils as mcu
import torch_xla.utils.utils as xu


def now(format='%H:%M:%S'):
  return datetime.now().strftime(format)

def _get_device_spec(device):
  ordinal = xm.get_ordinal(defval=-1)
  return str(device) if ordinal < 0 else '{}/{}'.format(device, ordinal)

def print_training_update(device,
                          step,
                          loss,
                          rate,
                          global_rate,
                          epoch=None):
  """Prints the training metrics at a given step.

  Args:
    device (torch.device): The device where these statistics came from.
    step_num (int): Current step number.
    loss (float): Current loss.
    rate (float): The examples/sec rate for the current batch.
    global_rate (float): The average examples/sec rate since training began.
    epoch (int, optional): The epoch number.
    summary_writer (SummaryWriter, optional): If provided, this method will
      write some of the provided statistics to Tensorboard.
  """
  update_data = [
      'Training', 'Device={}'.format(_get_device_spec(device)),
      'Epoch={}'.format(epoch) if epoch is not None else None,
      'Step={}'.format(step), 'Loss={:.5f}'.format(loss),
      'Rate={:.2f}'.format(rate), 'GlobalRate={:.2f}'.format(global_rate),
      'Time={}'.format(now())
  ]
  print('|', ' '.join(item for item in update_data if item), flush=True)



def print_test_update(device, accuracy, epoch=None, step=None):
  """Prints single-core test metrics.

  Args:
    device: Instance of `torch.device`.
    accuracy: Float.
  """
  update_data = [
      'Test', 'Device={}'.format(_get_device_spec(device)),
      'Step={}'.format(step) if step is not None else None,
      'Epoch={}'.format(epoch) if epoch is not None else None,
      'Accuracy={:.2f}'.format(accuracy) if accuracy is not None else None,
      'Time={}'.format(now())
  ]
  print('|', ' '.join(item for item in update_data if item), flush=True)
