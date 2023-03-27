# import torch, os
# from torch.utils.data import DataLoader
# from ..classes.datasets import IMDBDataset
# from transformers.models.bert import BertTokenizerFast
# import pandas as pd
# from ..utils.pickleUtils import pload

# cwd = os.path.split(__file__)[0]

# tokenizer:BertTokenizerFast = BertTokenizerFast.from_pretrained('bert-base-uncased') #type:ignore
# train_set: pd.DataFrame = pload(os.path.join(cwd, 'test_datasets/train_set'))
# train_labels: list[list] = train_set['sentiment'].tolist()
# train_texts: list[str] = train_set['review'].tolist()
# train_encodings = tokenizer(train_texts, truncation=True, padding=True)
# train_dataset = IMDBDataset(encodings=train_encodings, labels=train_labels)
