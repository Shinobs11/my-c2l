from torch import topk
import wandb, os, time
from preTraining import pretrainBERT
from evaluateModel import evaluateModel
from contrastiveLearning import constrastiveTrain
from generateTriplets import generateTiplets
import torch, gc
import wandb
import wandb.util as wbutil
import random



torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
os.environ['TOKENIZERS_PARALLELISM'] = "false"

def main():
  gc.collect()
  torch.cuda.empty_cache()
  
  dataset_name = "yelp_chi"
  # dataset_name = "original_augmented_1x_aclImdb"
  experiment_id = f"{dataset_name}_im_done"
  wandb.init(
    project="my-c2l",
    group=experiment_id,
    config={
      "dataset": dataset_name,
      "max_masking_attempts": 50,
      "batch_size_pt": 2,
      "n_epochs_pt": 2,
      "batch_size_cl": 2,
      "n_epochs_cl": 2,
      "batch_size_pt_eval": 24,
      "batch_size_cl_eval": 24,
    },
  )

    #pretrain
  
  def pt():
    # wandb.init(
    #     project="my-c2l",
    #     group=experiment_id,
    #     job_type="pretrain",
    #     reinit=True,
    #     config={
    #       "dataset": dataset_name,
    #       "batch_size_pt": 8,
    #       "n_epochs_pt":16,
    #     },
    #     magic=True
    #     )
    gc.collect()
    torch.cuda.empty_cache()
    pretrainBERT(
        dataset_name=wandb.config.dataset,
        batch_size=wandb.config.batch_size_pt,
        epoch_num=wandb.config.n_epochs_pt,
        use_pinned_memory=False,
        lr=wandb.config.lr,
    )
    # wandb.finish()

  pt()

  # def triplets():
  #   # triplets
  #   # wandb.init(
  #   #   project="my-c2l",
  #   #   group=experiment_id,
  #   #   job_type="generate_triplets",
  #   #   reinit=True,
  #   #   config={
  #   #    "dataset": dataset_name,
  #   #    "sampling_ratio": 1,
  #   #     "augment_ratio": 1,
  #   #     "dropout_ratio": 0.5,
  #   #     "max_masking_attempts": 50,
  #   #    },
  #   #    magic=True
  #   # )
  #   gc.collect()
  #   torch.cuda.empty_cache()
  #   generateTiplets(
  #       dataset_name=wandb.config.dataset,
  #       sampling_ratio=wandb.config.sampling_ratio,
  #       augment_ratio=wandb.config.augment_ratio,
  #       dropout_ratio=wandb.config.dropout_ratio,
  #       topk_num=wandb.config.topk_num,
  #       max_masking_attempts=wandb.config.max_masking_attempts,
  #       use_pinned_memory=False
  #   )
  #   # wandb.finish()

  # triplets()




  #   #contrastive training
  def cl():
    # wandb.init(
    #    project="my-c2l",
    #     group=experiment_id,
    #     job_type="cl_training",
    #     reinit=True,
    #     config={
    #    "dataset": dataset_name,
    #     "batch_size_cl": 2,
    #     "n_epochs_cl": 16
    #    },
    #    magic=True
    # )
    gc.collect()
    torch.cuda.empty_cache()
    constrastiveTrain(
        dataset_name=wandb.config.dataset,
        epoch_num=wandb.config.n_epochs_cl,
        lambda_weight=wandb.config.lambda_weight,
        batch_size=wandb.config.batch_size_cl,
        lr=wandb.config.lr,
    )
    wandb.finish()

  cl()

  def eval_pt():
    pretrain_eval_batch_size = 8
    pre_use_cl_model = False
    # wandb.init(
    #    project="my-c2l",
    #     group=experiment_id,
    #     job_type="pretrain_evaluation",
    #     reinit=True,
    #     config={
    #    "dataset": dataset_name,
    #    "batch_size_pt_eval": 24,
    #    },
    #    magic=True
    # )
    evaluateModel(
        dataset_name=wandb.config.dataset,
        batch_size=wandb.config.batch_size_pt_eval,
        use_cl_model=pre_use_cl_model
    )
    # wandb.finish()


  eval_pt()

  def eval_cl():
    cl_eval_batch_size = 8
    cl_use_cl_model = True
    # wandb.init(
    #    project="my-c2l",
    #     group=experiment_id,
    #     job_type="cl_evaluation",
    #     reinit=True,
    #     config={
    #    "dataset": dataset_name,
    #     "batch_size_cl_eval": 24,
    #    },
    #    magic=True
    # )
    evaluateModel(
        dataset_name=wandb.config.dataset,
        batch_size=wandb.config.batch_size_cl_eval,
        use_cl_model=cl_use_cl_model
    )
    # wandb.finish()

  eval_cl()
  
  time.sleep(15)
  gc.collect()
  torch.cuda.empty_cache()
if __name__=="__main__":
    torch.manual_seed(0)
    import numpy as np
    np.random.seed(0)
    import random
    random.seed(0)
    torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True)


    sweep_config = {
      'project': 'my-c2l',
      'method': 'bayes',
      'parameters':{
        'lr': {
          'distribution': 'log_uniform_values',
          'min': 1e-8,
          'max': 8e-5
        },
        'lambda_weight':{
          'distribution': 'uniform', 
          'min': 1e-12,
          'max': 1
        },
        'topk_num': {
          'distribution': 'int_uniform',
          'min': 1,
          'max': 10
        },
        'sampling_ratio': {
          'distribution': 'int_uniform',
          'min': 1,
          'max': 10
        },
        'augment_ratio': {
          'values': [1]
        },
        'dropout_ratio': {
          'distribution': 'uniform',
          'min': 0.0,
          'max': 1.0
        }
      },
      'metric': {
        'name': 'eval_accuracy_cl',
        'goal': 'maximize'
      }, 
    }
    sweep_id = wandb.sweep(sweep=sweep_config, project='my-c2l')
    wandb.agent(sweep_id=sweep_id, function=main, count=20)
    wandb.finish()