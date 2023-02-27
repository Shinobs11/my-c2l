import json, os, pickle, torch, logging, typing, numpy as np, glob, argparse, shutil
from preTraining import pretrainBERT
from generateTriplets import generateTiplets
from contrastiveLearning import constrastiveTrain
from evaluateModel import evaluateModel
parser = argparse.ArgumentParser(
  usage="python3 main.py [options...]",
  description="Implementation of Causally Contrastive Learning for Robust Text Classification from https://ojs.aaai.org/index.php/AAAI/article/download/21296/version/19583/21045 "
)


parser.add_argument(
  '--pretrain',
  action='store_true',
  default=None
)
parser.add_argument(
  '--generate-triplets',
  action='store_true',
  default=None
)
parser.add_argument(
  '--constrastive-train',
  action='store_true',
  default=None
)
parser.add_argument(
  '--evaluate',
  action='store_true',
  default=None
)
parser.add_argument(
  '--dataset-name',
  type=str,
  required=True
)
parser.add_argument(
  '--batch-size',
  type=int,
  required=True,
  default=16
)
parser.add_argument(
  '--epoch-num',
  type=int,
  required=True,
  default=15
)
parser.add_argument(
  '--use-margin-loss',
  action='store_true'
)
parser.add_argument(
  '--lambda-weight',
  type=float,
  default=0.1
)
parser.add_argument(
  '--use-cache',
  action='store_true'
)
parser.add_argument(
  '--use-pinned-memory', #TODOS: make sure this works
  action='store_true',
  default=False
)
parser.add_argument(
  '--clear-cache',
  action='store_true',
  default=False
)
parser.add_argument(
  '--clear-models',
  action='store_true',
  default=False
)
parser.add_argument(
  '--clear-base-models',
  action='store_true',
  default=False
)

args = parser.parse_args()

if args.clear_cache == True:
  for x in glob.glob("./cache/*"):
    shutil.rmtree(x)
if args.clear_models == True:
  for x in glob.glob("./models/*"):
    shutil.rmtree(x)
if args.clear_base_models == True:
  shutil.rmtree("~/.cache/huggingface/hub/models--bert-base-uncased")

if args.pretrain == True:
  pretrainBERT(
    dataset_name=args.dataset_name,
    batch_size=args.batch_size,
    epoch_num=args.epoch_num,
    use_margin_loss=args.use_margin_loss,
    lambda_weight=args.lambda_weight,
    use_cache=args.use_cache,
    use_pinned_memory=args.use_pinned_memory
  )
if args.generate_triplets == True:
  generateTiplets(
    dataset_name=args.dataset_name,
    batch_size=args.batch_size,
    epoch_num=args.epoch_num,
    use_margin_loss=args.use_margin_loss,
    lambda_weight=args.lambda_weight,
    use_cache=args.use_cache,
    use_pinned_memory=args.use_pinned_memory
  )
if args.constrastive_train == True:
  constrastiveTrain(
    dataset_name=args.dataset_name,
    batch_size=args.batch_size,
    epoch_num=args.epoch_num,
    use_margin_loss=args.use_margin_loss,
    lambda_weight=args.lambda_weight,
    use_cache=args.use_cache
  )
if args.evaluate == True:
  evaluateModel(
    dataset_name=args.dataset_name,
    batch_size=args.batch_size,
    epoch_num=args.epoch_num,
    use_margin_loss=args.use_margin_loss,
    lambda_weight=args.lambda_weight,
    use_cache=args.use_cache
  )