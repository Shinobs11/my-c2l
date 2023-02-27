from src.utils.pickleUtils import pload, pdump, pjoin
import os, json, argparse, glob

import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument(
  '--dataset-path',
  type=str,
  required=True
)
args = parser.parse_args()

DATASET_PATH = args.dataset_path

def convertToTable(items:list[dict])-> pd.DataFrame:
  
  table_dict:dict[str, list] = {
    "anchor_text": [],
    "label": []
  }

  for x in items:
    for k in table_dict.keys():
      table_dict[k].append(x[k])
  
  table_dict["label"] = [reversed(l) for l in table_dict["label"]]
  table = pd.DataFrame({
    "text": table_dict["anchor_text"],
    "label": [[x for x in l] for l in table_dict["label"]]
  })
  # table["label"] = table["label"].transform(lambda x: 0 if x == [0.0, 1.0] else 1)

  return table
  

if(os.path.isdir(DATASET_PATH)):
  paths = glob.glob(os.path.join(DATASET_PATH, "*.json"))
  for p in paths:
    with open(p, mode='r') as f:
      table = convertToTable(json.load(f))
      pdump(table, os.path.splitext(p)[0])
      table.to_csv(os.path.splitext(p)[0] + ".csv")
      
else:
  with open(DATASET_PATH, mode='r') as f:
    table = convertToTable(json.load(f))
    pdump(table, os.path.splitext(DATASET_PATH)[0])
    table.to_csv(os.path.splitext(DATASET_PATH)[0] + ".csv")

