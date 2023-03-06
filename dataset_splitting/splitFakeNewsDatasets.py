import pandas as pd
import os, numpy as np
from src.utils.pickleUtils import pload, pdump


MAX_ROWS = 2000
NEWS_PATH = "datasets/fake_news_1"
real_news = pd.read_csv(os.path.join(NEWS_PATH, "True.csv"))
fake_news = pd.read_csv(os.path.join(NEWS_PATH, "Fake.csv"))


real_news["fake"] = [[1.0, 0.0] for x in range(len(real_news))]
fake_news["fake"] = [[0.0, 1.0] for x in range(len(fake_news))]

pre_dataset = pd.concat([real_news, fake_news]).sample(frac=1).reset_index(drop=True)[0:MAX_ROWS]
dataset = pd.DataFrame()
dataset["label"] = pre_dataset["fake"]
dataset["text"] = pre_dataset["title"] + "\n" + pre_dataset["text"]

  



train_set = dataset[:int(len(dataset)*0.75)]
test_set = dataset[int(len(dataset)*0.75):int(len(dataset)*0.95)]
valid_set = dataset[int(len(dataset)*0.95):]

if not os.path.exists(os.path.join(NEWS_PATH, "base")):
  os.makedirs(os.path.join(NEWS_PATH, "base"))

BASE = os.path.join(NEWS_PATH, "base")
train_set.to_csv(os.path.join(BASE, "train_set.csv"), index=False)
test_set.to_csv(os.path.join(BASE, "test_set.csv"), index=False)
valid_set.to_csv(os.path.join(BASE, "valid_set.csv"), index=False)

pdump(train_set, os.path.join(BASE, "train_set"))
pdump(test_set, os.path.join(BASE, "test_set"))
pdump(valid_set, os.path.join(BASE, "valid_set"))
