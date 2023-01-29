MAX_ROWS = 100
MAX_ROWS = int(100/0.65)


DATASET = "imdb"
DATASET_PATH = f"datasets/{DATASET}"


from random import shuffle
import numpy as np
import pandas as pd
import os, pickle, blosc, glob



for f in glob.glob("./datasets/imdb/base/*.pickle.blosc"):
  os.remove(f)



csv = pd.read_csv("./datasets/imdb/dataset.csv")
cols = list(csv.columns)
cols.reverse()
csv = csv.reindex(columns=cols)
csv = csv.drop(csv.tail(-MAX_ROWS).index)



r = np.random.choice(csv.index, len(csv), replace=False)


train_set = csv.drop(list(r[int(len(r)*0.65):]))
val_set = csv.drop(list(r[int(len(r)*0.65):int(len(r)*0.75)]))
test_set = csv.drop(list(r[0:int(len(r)*0.75)]))

train_set.reset_index(drop=True, inplace=True)
val_set.reset_index(drop=True, inplace=True)
test_set.reset_index(drop=True, inplace=True)




tokenized_train_set = pd.DataFrame({
  "sentiment": train_set['sentiment'].transform(lambda x: [1, 0] if str(x).lower() == 'positive' else [0, 1]),
  "review": train_set['review']
})
tokenized_val_set = pd.DataFrame({
  "sentiment": val_set['sentiment'].transform(lambda x: [1, 0] if str(x).lower() == 'positive' else [0, 1]),
  "review": val_set['review']
})
tokenized_test_set = pd.DataFrame({
  "sentiment": test_set['sentiment'].transform(lambda x: [1, 0] if str(x).lower() == 'positive' else [0, 1]),
  "review": test_set['review']
})





basepath: str = os.path.join(DATASET_PATH, "base")

if not os.path.exists(basepath):
  os.mkdir(basepath)

train_set.to_csv(os.path.join(basepath,"train_set.csv"), index=False)
val_set.to_csv(os.path.join(basepath,"val_set.csv"), index=False)
test_set.to_csv(os.path.join(basepath,"test_set.csv"), index=False)


with open(os.path.join(basepath,"train_set.pickle.blosc"), mode='wb') as f:
  f.write(blosc.compress(pickle.dumps(tokenized_train_set)))

with open(os.path.join(basepath,"val_set.pickle.blosc"), mode='wb') as f:
  f.write(blosc.compress(pickle.dumps(tokenized_val_set)))

with open(os.path.join(basepath,"test_set.pickle.blosc"), mode='wb') as f:
  f.write(blosc.compress(pickle.dumps(tokenized_test_set)))