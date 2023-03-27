# import json, os, pickle, torch, logging, typing, numpy as np, glob, argparse, shutil
# from preTraining import pretrainBERT
# from generateTriplets import generateTiplets
# from contrastiveLearning import constrastiveTrain
# from evaluateModel import evaluateModel
# import torch_xla.core.xla_model as xm

# torch.manual_seed(0)
# import numpy as np
# np.random.seed(0)
# import random
# random.seed(0)


# if torch.cuda.is_available():
#   os.environ['PJRT_DEVICE'] = 'GPU'
# elif xm.get_xla_supported_devices('TPU') is not None:
#   os.environ['PJRT_DEVICE'] = 'TPU'
# else:
#   os.environ['PJRT_DEVICE'] = 'CPU'

# parser = argparse.ArgumentParser(
#   usage="python3 main.py [options...]",
#   description="Implementation of Causally Contrastive Learning for Robust Text Classification from https://ojs.aaai.org/index.php/AAAI/article/download/21296/version/19583/21045 "
# )


# parser.add_argument(
#   '--pretrain',
#   action='store_true',
#   default=None
# )
# parser.add_argument(
#   '--generate-triplets',
#   action='store_true',
#   default=None
# )
# parser.add_argument(
#   '--constrastive-train',
#   action='store_true',
#   default=None
# )
# parser.add_argument(
#   '--evaluate',
#   action='store_true',
#   default=None
# )
# parser.add_argument(
#   '--dataset-name',
#   type=str,
# )
# parser.add_argument(
#   '--batch-size',
#   type=int,
#   default=16
# )
# parser.add_argument(
#   '--epoch-num',
#   type=int,
#   default=15
# )

# parser.add_argument(
#   '--lambda-weight',
#   type=float,
#   default=0.1
# )
# #triplets args
# parser.add_argument(
#   '--max-masking-attempts',
#   type=int,
#   default=0
# )
# parser.add_argument(
#   '--sampling-ratio',
#   type=int,
#   default=1
# )
# parser.add_argument(
#   '--augment-ratio',
#   type=int,
#   default=1
# )
# parser.add_argument(
#   '--topk-num',
#   type=int,
#   default=4
# )
# parser.add_argument(
#   '--dropout-ratio',
#   type=float,
#   default=0.5
# )

# parser.add_argument(
#   '--use-cl-model',
#   action='store_true',
# )

# parser.add_argument(
#   '--use-cache',
#   action='store_true'
# )
# parser.add_argument(
#   '--use-pinned-memory', #TODOS: make sure this works
#   action='store_true',
#   default=False
# )
# parser.add_argument(
#   '--clear-cache',
#   action='store_true',
#   default=False
# )
# parser.add_argument(
#   '--clear-models',
#   action='store_true',
#   default=False
# )
# parser.add_argument(
#   '--clear-base-models',
#   action='store_true',
#   default=False
# )

# args = parser.parse_args()

# if args.clear_cache == True:
#   for x in glob.glob("./cache/*"):
#     shutil.rmtree(x)
# if args.clear_models == True:
#   for x in glob.glob("./models/*"):
#     shutil.rmtree(x)
# if args.clear_base_models == True:
#   shutil.rmtree("~/.cache/huggingface/hub/models--bert-base-uncased")





# if args.pretrain == True:
#   pretrainBERT(
#     dataset_name=args.dataset_name,
#     batch_size=args.batch_size,
#     epoch_num=args.epoch_num,
#     use_pinned_memory=args.use_pinned_memory
#   )
# if args.generate_triplets == True:
#   generateTiplets(
#     dataset_name=args.dataset_name,
#     use_pinned_memory=args.use_pinned_memory,
#     max_masking_attempts=args.max_masking_attempts,
#     topk_num=args.topk_num,
#     sampling_ratio=args.sampling_ratio,
#     augment_ratio=args.augment_ratio,
#     dropout_ratio=args.dropout_ratio
#   )
# if args.constrastive_train == True:
#   constrastiveTrain(
#     dataset_name=args.dataset_name,
#     batch_size=args.batch_size,
#     epoch_num=args.epoch_num,
#     lambda_weight=args.lambda_weight
#   )
# if args.evaluate == True:
#   evaluateModel(
#     dataset_name=args.dataset_name,
#     batch_size=args.batch_size,
#     use_cl_model=args.use_cl_model
#   )





