from torch import topk
import wandb, os
from preTraining import pretrainBERT
from evaluateModel import evaluateModel 
from contrastiveLearning import constrastiveTrain #TODOS: figure out why these imports break EVERYTHING
from generateTriplets import generateTiplets
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch
torch.manual_seed(0)
import numpy as np
np.random.seed(0)
import random
random.seed(0)
import logging
from src.logging.logSetup import logSetup
import wandb
import wandb.util as wbutil
import random

from src.arg_classes import Args_PT



# if torch.cuda.is_available():
#   os.environ['PJRT_DEVICE'] = 'GPU'
# elif xm.get_xla_supported_devices('TPU') is not None:
#   os.environ['PJRT_DEVICE'] = 'TPU'
# else:
#   os.environ['PJRT_DEVICE'] = 'CPU'
# os.environ['PJRT_DEVICE'] = 'TPU'
os.environ['TOKENIZERS_PARALLELISM'] = "false"
# os.environ['XRT_TPU_CONFIG'] = 'localservice;0;localhost:51011'
# os.environ['MASTER_ADDR'] = 'localhost'
# os.environ['MASTER_PORT'] = '12355'


def main():

    # print("Entered main")
    dataset_name = "yelp_nyc"
    # experiment_id = f"{dataset_name}-experiment-{wbutil.generate_id()}"

    #pretrain
  


    # wandb.init(
    #     project="my-c2l",
    #     group=experiment_id,
    #     job_type="pretrain",
    #     reinit=True,
    #     config={
    #       "dataset": dataset_name,
    #       "batch_size_pt": 8,
    #       "n_epochs_pt":20,
    #     }
    #     )
    

    # pretrainBERT(
    #     dataset_name=wandb.config.dataset,
    #     batch_size=wandb.config.batch_size_pt,
    #     epoch_num=wandb.config.n_epochs_pt,
    #     use_pinned_memory=False,
    #     use_wandb=False
    # )

    args_pt = Args_PT(
        dataset_name=dataset_name,
        batch_size = 8,
        epoch_num = 20,
        use_pinned_memory=False,
        use_wandb=False,
        num_workers=4)


    pretrainBERT(
      args_pt
    )
    # wandb.finish()


    # #triplets
    # wandb.init(
    #   project="my-c2l",
    #   group=experiment_id,
    #   job_type="generate_triplets",
    #   reinit=True,
    #   config={
    #    "dataset": dataset_name,
    #    "sampling_ratio": 1,
    #     "augment_ratio": 1,
    #     "dropout_ratio": 0.5,
    #     "topk_num": 4,
    #     "max_masking_attempts": 0,
    #    }
    # )
    # generateTiplets(
    #     dataset_name=wandb.config.dataset,
    #     sampling_ratio=wandb.config.sampling_ratio,
    #     augment_ratio=wandb.config.augment_ratio,
    #     dropout_ratio=wandb.config.dropout_ratio,
    #     topk_num=wandb.config.topk_num,
    #     max_masking_attempts=wandb.config.max_masking_attempts,
    #     use_pinned_memory=False
    # )
    # wandb.finish()

    # #contrastive training

    # wandb.init(
    #    project="my-c2l",
    #     group=experiment_id,
    #     job_type="cl_training",
    #     reinit=True,
    #     config={
    #    "dataset": dataset_name,
    #     "batch_size_cl": 8,
    #     "n_epochs_cl": 20,
    #     "lambda_weight": 0.1,
    #    }
    # )
    # constrastiveTrain(
    #     dataset_name=wandb.config.dataset,
    #     epoch_num=wandb.config.n_epochs_cl,
    #     lambda_weight=wandb.config.lambda_weight,
    #     batch_size=wandb.config.batch_size_cl
    # )
    # wandb.finish()

    # #eval(pretrain)
    # pretrain_eval_batch_size = 8
    # pre_use_cl_model = False
    # wandb.init(
    #    project="my-c2l",
    #     group=experiment_id,
    #     job_type="pretrain_evaluation",
    #     reinit=True,
    #     config={
    #    "dataset": dataset_name,
    #    "batch_size_pt_eval": 32,
    #    }
    # )
    # evaluateModel(
    #     dataset_name=wandb.config.dataset,
    #     batch_size=wandb.config.batch_size_pt_eval,
    #     use_cl_model=pre_use_cl_model
    # )
    # wandb.finish()
    # #eval(cl)
    # cl_eval_batch_size = 8
    # cl_use_cl_model = True
    # wandb.init(
    #    project="my-c2l",
    #     group=experiment_id,
    #     job_type="cl_evaluation",
    #     reinit=True,
    #     config={
    #    "dataset": dataset_name,
    #     "batch_size_cl_eval": 32,
    #    }
    # )
    # evaluateModel(
    #     dataset_name=wandb.config.dataset,
    #     batch_size=wandb.config.batch_size_cl_eval,
    #     use_cl_model=cl_use_cl_model
    # )
    # wandb.finish()

# def test_fn(index, args):
#   xm.master_print(f"MasterPrint: Started process on index {index}")
#   xm.master_print(f"My name is {__name__}")
#   print(f"Started process on index {index}")
#   log = logSetup(f"logs/pt/{args.dataset_name}_{index}", f"logs/pt/{args.dataset_name}_{index}.log", logging.INFO, None)
#   log.info(f"Started process on index {index}")
#   device = xm.xla_device()
#   log.info(f"Device: {device}")
  
#   test_tensor = torch.tensor([1,2,3,4,5]).to(device)
#   test_tensor = test_tensor.sum()
#   log.info(f"Test tensor post-sum: {test_tensor}")
#   res_tensor = xm.all_reduce(xm.REDUCE_SUM, [test_tensor], scale=1.0)
#   log.info(f"Result tensor: {res_tensor}")


if __name__ == '__main__':
  print(f"My name is {__name__}")
  # xmp.spawn(main, nprocs=1, start_method='spawn')
  # args_pt = Args_PT(
  #   dataset_name="yelp_nyc",
  #   batch_size=8,
  #   epoch_num=20,
  #   use_pinned_memory=False,
  #   use_wandb=False
  # )

  # xmp.spawn(test_fn, args=(args_pt,), start_method='spawn', join=True)



  main()