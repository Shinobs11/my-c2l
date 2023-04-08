import sys, os
sys.path.append(f"{os.getcwd()}")
from operator import index
import pandas as pd
import numpy as np
from src.utils.pickleUtils import pdump, pload
import os


#for CHI, review being fake is Y or 0, N or 1

TRAIN_SIZE = 400
TEST_SIZE = 100

yelp_chi_train = pd.read_csv("datasets/yelp_chi/yelp_chi_train.csv", index_col = 0).reset_index(drop=True)[0:TRAIN_SIZE]
yelp_chi_test = pd.read_csv("datasets/yelp_chi/yelp_chi_test.csv", index_col = 0).reset_index(drop=True)[0:TEST_SIZE]
yelp_nyc_train = pd.read_csv("datasets/yelp_nyc/yelp_nyc_train.csv", index_col = 0).reset_index(drop=True)[0:TRAIN_SIZE]
yelp_nyc_test = pd.read_csv("datasets/yelp_nyc/yelp_nyc_test.csv", index_col = 0).reset_index(drop=True)[0:TEST_SIZE]
yelp_zip_train = pd.read_csv("datasets/yelp_zip/yelp_zip_train.csv", index_col = 0).reset_index(drop=True)[0:TRAIN_SIZE]
yelp_zip_test = pd.read_csv("datasets/yelp_zip/yelp_zip_test.csv", index_col = 0).reset_index(drop=True)[0:TEST_SIZE]

yelp_chi_train = yelp_chi_train.drop(columns=["review_id", "reviewer_id", "product_id", "date", "star", "label"])
yelp_chi_test = yelp_chi_test.drop(columns=["review_id", "reviewer_id", "product_id", "date", "star", "label"])
yelp_nyc_train = yelp_nyc_train.drop(columns=["review_id", "business_id", "date", "stars", "label"])
yelp_nyc_test = yelp_nyc_test.drop(columns=["review_id", "business_id", "date", "stars", "label"])
yelp_zip_train = yelp_zip_train.drop(columns=["review_id", "business_id", "date", "stars", "label"])
yelp_zip_test = yelp_zip_test.drop(columns=["review_id", "business_id", "date", "stars", "label"])



yelp_chi_train = yelp_chi_train.rename(columns={"content": "text", "label_new": "label"})
yelp_chi_test = yelp_chi_test.rename(columns={"content": "text", "label_new": "label"})
yelp_nyc_train = yelp_nyc_train.rename(columns={"content": "text", "label_new": "label"})
yelp_nyc_test = yelp_nyc_test.rename(columns={"content": "text", "label_new": "label"})
yelp_zip_train = yelp_zip_train.rename(columns={"content": "text", "label_new": "label"})
yelp_zip_test = yelp_zip_test.rename(columns={"content": "text", "label_new": "label"})

#[1.0, 0.0] is fake
#[0.0, 1.0] is not fake

yelp_chi_train["label"] = yelp_chi_train["label"].transform((lambda x: [0.0, 1.0] if x == 1 else [1.0, 0.0]))
yelp_chi_test["label"] = yelp_chi_test["label"].transform((lambda x: [0.0, 1.0] if x == 1 else [1.0, 0.0]))
yelp_nyc_train["label"] = yelp_nyc_train["label"].transform((lambda x: [0.0, 1.0] if x == 1 else [1.0, 0.0]))
yelp_nyc_test["label"] = yelp_nyc_test["label"].transform((lambda x: [0.0, 1.0] if x == 1 else [1.0, 0.0]))
yelp_zip_train["label"] = yelp_zip_train["label"].transform((lambda x: [0.0, 1.0] if x == 1 else [1.0, 0.0]))
yelp_zip_test["label"] = yelp_zip_test["label"].transform((lambda x: [0.0, 1.0] if x == 1 else [1.0, 0.0]))


chi_train_set = yelp_chi_train.drop(yelp_chi_train.index[int(len(yelp_chi_train)*0.75):])
chi_valid_set = yelp_chi_train.drop(yelp_chi_train.index[:int(len(yelp_chi_train)*0.75)])
chi_test_set = yelp_chi_test

nyc_train_set = yelp_nyc_train.drop(yelp_nyc_train.index[int(len(yelp_nyc_train)*0.75):])
nyc_valid_set = yelp_nyc_train.drop(yelp_nyc_train.index[:int(len(yelp_nyc_train)*0.75)])
nyc_test_set = yelp_nyc_test

zip_train_set = yelp_zip_train.drop(yelp_zip_train.index[int(len(yelp_zip_train)*0.75):])
zip_valid_set = yelp_zip_train.drop(yelp_zip_train.index[:int(len(yelp_zip_train)*0.75)])
zip_test_set = yelp_zip_test



CHI_PATH = "datasets/yelp_chi"
NYC_PATH = "datasets/yelp_nyc"
ZIP_PATH = "datasets/yelp_zip"


for p in (CHI_PATH, NYC_PATH, ZIP_PATH):
  if not os.path.exists(os.path.join(p, "base")):
    os.makedirs(os.path.join(p, "base"))

data_sets = (
  (chi_train_set, chi_valid_set, chi_test_set, os.path.join(CHI_PATH, "base")),
  (nyc_train_set, nyc_valid_set, nyc_test_set, os.path.join(NYC_PATH, "base")),
  (zip_train_set, zip_valid_set, zip_test_set, os.path.join(ZIP_PATH, "base"))
)



for ds in data_sets:
  pdump(ds[0], os.path.join(ds[3], "train_set"))
  pdump(ds[1], os.path.join(ds[3], "valid_set"))
  pdump(ds[2], os.path.join(ds[3], "test_set"))

  ds[0].to_csv(os.path.join(ds[3], "train_set.csv"), index=False)
  ds[1].to_csv(os.path.join(ds[3], "valid_set.csv"), index=False)
  ds[2].to_csv(os.path.join(ds[3], "test_set.csv"), index=False)
