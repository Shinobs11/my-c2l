from torch import topk
import wandb
from preTraining import pretrainBERT
from evaluateModel import evaluateModel
from contrastiveLearning import constrastiveTrain
from generateTriplets import generateTiplets

import torch
torch.manual_seed(0)
import numpy as np
np.random.seed(0)
import random
random.seed(0)

import wandb
import random



def main():
    wandb.init(
      project="my-c2l",
      config={
        "dataset": "original_augmented_1x_aclImdb",
        "pre_epoch_num": 5,
        "sampling_ratio": 1,
        "augment_ratio": 1,
        "dropout_ratio": 1,
        "topk_num": 1,
        "max_masking_attempts": 0,
        "cl_epoch_num": 5,
        "lambda_weight": 0.1
      }
    )
    dataset_name = "original_augmented_1x_aclImdb"
    

    #pretrain
    pretrain_batch_size = 8
    pretrain_epoch_num = 5

    pretrainBERT(
        dataset_name=dataset_name,
        batch_size=pretrain_batch_size,
        epoch_num=pretrain_epoch_num,
        use_pinned_memory=False
    )



    #triplets
    sampling_ratio = 1
    augment_ratio = 1
    dropout_ratio = 0.5
    topk_num = 4
    max_masking_attempts = 0
    generateTiplets(
        dataset_name=dataset_name,
        sampling_ratio=sampling_ratio,
        augment_ratio=augment_ratio,
        dropout_ratio=dropout_ratio,
        topk_num=topk_num,
        max_masking_attempts=max_masking_attempts,
        use_pinned_memory=False
    )


    #contrastive training
    contrastive_learning_batch_size = 2
    cl_epoch_num = 5
    lambda_weight = 0.1

    constrastiveTrain(
        dataset_name=dataset_name,
        epoch_num=cl_epoch_num,
        lambda_weight=lambda_weight,
        batch_size=contrastive_learning_batch_size
    )


    #eval(pretrain)
    pretrain_eval_batch_size = 8
    pre_use_cl_model = False

    evaluateModel(
        dataset_name=dataset_name,
        batch_size=pretrain_eval_batch_size,
        use_cl_model=pre_use_cl_model
    )

    #eval(cl)
    cl_eval_batch_size = 8
    cl_use_cl_model = True

    evaluateModel(
        dataset_name=dataset_name,
        batch_size=cl_eval_batch_size,
        use_cl_model=cl_use_cl_model
    )
    wandb.finish()

if __name__=="__main__":
    main()