from torch import topk
import wandb, os
from preTraining import pretrainBERT
from evaluateModel import evaluateModel
from contrastiveLearning import constrastiveTrain
from generateTriplets import generateTiplets
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch
torch.manual_seed(0)
import numpy as np
np.random.seed(0)
import random
random.seed(0)

import wandb
import wandb.util as wbutil
import random

if torch.cuda.is_available():
  os.environ['PJRT_DEVICE'] = 'GPU'
elif xm.get_xla_supported_devices('TPU') is not None:
  os.environ['PJRT_DEVICE'] = 'TPU'
else:
  os.environ['PJRT_DEVICE'] = 'CPU'

os.environ['TOKENIZERS_PARALLELISM'] = "false"

def main():

    dataset_name = "yelp_nyc"
    experiment_id = f"{dataset_name}-experiment-{wbutil.generate_id()}"

    #pretrain
  

    wandb.init(
        project="my-c2l",
        group=experiment_id,
        job_type="pretrain",
        reinit=True,
        config={
          "dataset": dataset_name,
          "batch_size_pt":128,
          "n_epochs_pt":20,
        }
        )
    pretrainBERT(
        dataset_name=wandb.config.dataset,
        batch_size=wandb.config.batch_size_pt,
        epoch_num=wandb.config.n_epochs_pt,
        use_pinned_memory=False
    )
    wandb.finish()


    #triplets
    wandb.init(
      project="my-c2l",
      group=experiment_id,
      job_type="generate_triplets",
      reinit=True,
      config={
       "dataset": dataset_name,
       "sampling_ratio": 1,
        "augment_ratio": 1,
        "dropout_ratio": 0.5,
        "topk_num": 4,
        "max_masking_attempts": 0,
       }
    )
    generateTiplets(
        dataset_name=wandb.config.dataset,
        sampling_ratio=wandb.config.sampling_ratio,
        augment_ratio=wandb.config.augment_ratio,
        dropout_ratio=wandb.config.dropout_ratio,
        topk_num=wandb.config.topk_num,
        max_masking_attempts=wandb.config.max_masking_attempts,
        use_pinned_memory=False
    )
    wandb.finish()

    #contrastive training

    wandb.init(
       project="my-c2l",
        group=experiment_id,
        job_type="cl_training",
        reinit=True,
        config={
       "dataset": dataset_name,
        "batch_size_cl": 16,
        "n_epochs_cl": 20,
        "lambda_weight": 0.1,
       }
    )
    constrastiveTrain(
        dataset_name=wandb.config.dataset,
        epoch_num=wandb.config.n_epochs_cl,
        lambda_weight=wandb.config.lambda_weight,
        batch_size=wandb.config.batch_size_cl
    )
    wandb.finish()

    #eval(pretrain)
    pretrain_eval_batch_size = 8
    pre_use_cl_model = False
    wandb.init(
       project="my-c2l",
        group=experiment_id,
        job_type="pretrain_evaluation",
        reinit=True,
        config={
       "dataset": dataset_name,
       "batch_size_pt_eval": 1024,
       }
    )
    evaluateModel(
        dataset_name=wandb.config.dataset,
        batch_size=wandb.config.batch_size_pt_eval,
        use_cl_model=pre_use_cl_model
    )
    wandb.finish()
    #eval(cl)
    cl_eval_batch_size = 8
    cl_use_cl_model = True
    wandb.init(
       project="my-c2l",
        group=experiment_id,
        job_type="cl_evaluation",
        reinit=True,
        config={
       "dataset": dataset_name,
        "batch_size_cl_eval": 1024,
       }
    )
    evaluateModel(
        dataset_name=wandb.config.dataset,
        batch_size=wandb.config.batch_size_cl_eval,
        use_cl_model=cl_use_cl_model
    )
    wandb.finish()

if __name__=="__main__":
    main()