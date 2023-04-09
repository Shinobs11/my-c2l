from hmac import new
import sys, os
sys.path.append(f"{os.getcwd()}")

from  src.utils.pickleUtils import pload, pdump, pjoin
import os, json, argparse, glob, typing
import ast
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument(
  '--dataset-path',
  type=str,
  required=True
)
args = parser.parse_args()

DATASET_PATH = args.dataset_path

def convertToTable(items:typing.List[dict])-> pd.DataFrame:
  
  table_dict:typing.Dict[str, list] = {
    "anchor_text": [],
    "label": []
  }

  for x in items:
    for k in table_dict.keys():
      table_dict[k].append(x[k])
  
 
  table_dict["label"] = [reversed(l) for l in table_dict["label"]] #to ensure labels conform with readme
  table = pd.DataFrame({
    "text": table_dict["anchor_text"],
    "label": [[l for l in lst] for lst in table_dict["label"]] #type:ignore
  })
  table['text'].transform(lambda x: x.strip('\"'))
  
  # table["label"] = table["label"].transform(lambda x: 0 if x == [0.0, 1.0] else 1)

  return table
import csv
if not os.path.exists(os.path.join(DATASET_PATH, "base")):
  os.makedirs(os.path.join(DATASET_PATH, "base"))
if(os.path.isdir(DATASET_PATH)):
  paths = glob.glob(os.path.join(DATASET_PATH, "*.json"))
  for p in paths:
    with open(p, mode='r') as f:
      table = convertToTable(json.load(f))
      path_no_ext = os.path.splitext(p)[0]
      print(path_no_ext)
      new_path = os.path.join(os.path.split(p)[0], "base", os.path.split(path_no_ext)[1])
      print(new_path)
      table.to_csv(new_path + ".csv", quoting=csv.QUOTE_ALL, index=False)
      
else:
  with open(DATASET_PATH, mode='r') as f:
    table = convertToTable(json.load(f))
    table.to_csv(os.path.splitext(DATASET_PATH)[0] + ".csv", quoting=csv.QUOTE_ALL, index=False)

