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
from split_datasets import split_datasets
from train_model import train_model
from src.configs.deepspeed_config import deepspeed_config
from copy import deepcopy
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
os.environ['TOKENIZERS_PARALLELISM'] = "false"
dataset_name = "yelp_nyc"
  # dataset_name = "original_augmented_1x_aclImdb"
def main():
  
  gc.collect()
  torch.cuda.empty_cache()
  
  modded_deepspeed_config = deepcopy(deepspeed_config)
  # modded_deepspeed_config['optimizer']['params']['lr'] = 5e-5
  wandb.init(
  project="my-c2l",
  config={
    "dataset": dataset_name,
    "max_masking_attempts": 50,
    "batch_size_pt": 16,
    "n_epochs_pt": 16,
    "batch_size_cl": 16,
    "n_epochs_cl": 16,
    "batch_size_pt_eval": 24,
    "batch_size_cl_eval": 24,
    "augment_ratio": 1
  },
)
   
  
  

  def base():
    gc.collect()
    torch.cuda.empty_cache()
    train_model(
      dataset_name=wandb.config.dataset,
      batch_size=8,
      epoch_num=wandb.config.n_epochs_pt,
      use_pinned_memory=True,
      lambda_weight=wandb.config.lambda_weight,
      lr=wandb.config.lr,
      use_cl=False,
      use_wandb=True,
      ds_config = modded_deepspeed_config
  )

  base()

  def triplets():
    gc.collect()
    torch.cuda.empty_cache()
    generateTiplets(
        dataset_name=dataset_name,
        sampling_ratio=wandb.config.sampling_ratio,
        augment_ratio=wandb.config.augment_ratio,
        dropout_ratio=wandb.config.dropout_ratio,
        topk_num=wandb.config.topk_num,
        max_masking_attempts=wandb.config.max_masking_attempts,
        use_pinned_memory=False,
    )


  triplets()

  def cl():
    gc.collect()
    torch.cuda.empty_cache()
    train_model(
      dataset_name=dataset_name,
      batch_size=wandb.config.batch_size_cl,
      epoch_num=wandb.config.n_epochs_cl,
      use_pinned_memory=True,
      lr=wandb.config.lr,
      lambda_weight=wandb.config.lambda_weight,
      use_cl=True,
      use_wandb=True,
      ds_config = modded_deepspeed_config 
    )

  cl()


  def eval_pt():
    pretrain_eval_batch_size = 8
    pre_use_cl_model = False
    evaluateModel(
        dataset_name=dataset_name,
        batch_size=wandb.config.batch_size_pt_eval,
        use_cl_model=pre_use_cl_model
    )



  eval_pt()

  def eval_cl():
    cl_eval_batch_size = 8
    cl_use_cl_model = True
    evaluateModel(
        dataset_name=dataset_name,
        batch_size=wandb.config.batch_size_cl_eval,
        use_cl_model=cl_use_cl_model
    )

  eval_cl()
  
  time.sleep(5)
  gc.collect()
  torch.cuda.empty_cache()
if __name__=="__main__":
  
  
    TOKEN_LIMIT = 128
    TRAIN_SIZE = 1200
    TEST_SIZE = 800
    DATASET_PATH = f"./datasets/{dataset_name}"
    if os.path.exists(os.path.join(DATASET_PATH, "dataset_meta.pt")):
      dataset_meta = torch.load(os.path.join(DATASET_PATH, "dataset_meta.pt"))
      if  dataset_meta["token_limit"] != TOKEN_LIMIT or \
          dataset_meta["train_size"] != TRAIN_SIZE or \
          dataset_meta["test_size"] != TEST_SIZE:
            split_datasets(train_size=TRAIN_SIZE, test_size=TEST_SIZE, token_limit=TOKEN_LIMIT)
    else:
      split_datasets(train_size=TRAIN_SIZE, test_size=TEST_SIZE, token_limit=TOKEN_LIMIT)
  
    import torch._dynamo as dyn
    dyn.reset()
    dyn.reset()
    gc.collect()
    torch.cuda.empty_cache()

    torch.manual_seed(0)
    import numpy as np
    np.random.seed(0)
    import random
    random.seed(0)
    torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True)

    # main()
    sweep_config = {
      'project': 'my-c2l',
      'method': 'bayes',
      'name': f'{dataset_name}_sweep_0',
      'parameters':{
        'lr': {
          'distribution': 'log_uniform_values',
          'min': 5e-8,
          'max': 1e-4
        },
        'lambda_weight':{
          'distribution': 'log_uniform_values', 
          'min': 1e-6,
          'max': 1
        },
        'topk_num': {
          'distribution': 'int_uniform',
          'min': 1,
          'max': 16
        },
        'sampling_ratio': {
          'distribution': 'int_uniform',
          'min': 1,
          'max': 16
        },
        'dropout_ratio': {
          'distribution': 'uniform',
          'min': 0.0,
          'max': 1.0
        }
      },
      'metric': {
        'name': 'cl_eval_accuracy',
        'goal': 'maximize'
      }, 
    }
    sweep_id = wandb.sweep(sweep=sweep_config, project='my-c2l')
    wandb.agent(sweep_id=sweep_id, function=main, count=256)
    wandb.finish()