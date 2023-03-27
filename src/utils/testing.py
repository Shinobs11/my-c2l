# import torch, os
# from torch import Tensor
# from pickleUtils import pload, pdump
# from torch.utils.data import DataLoader
# import transformers as T
# from transformers import BertTokenizerFast, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
# DATASET_NAME = "imdb"
# DATASET_PATH = f"./datasets/{DATASET_NAME}/base"
# OUTPUT_PATH = f"checkpoints/{DATASET_NAME}/model"




# def test():
#   tokenizer: T.BertTokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
#   train_set = pload(os.path.join(DATASET_PATH, "train_set"))
#   train_texts:list[str] = train_set['review'].tolist()
#   train_labels:list = train_set['sentiment'].tolist()
#   train_encodings = pload("datasets/imdb/base/train_encodings")

#   from datasetClasses import IMDBDataset
#   train_dataset = IMDBDataset(labels=train_labels, encodings=train_encodings)

#   train_loader = DataLoader(
#     train_dataset,
#     batch_size=1,
#     shuffle=False,
#     persistent_workers=False, ##Switch to true if dataloader is used multiple times
#     pin_memory=True)


#   averageImportances = pload(os.path.join(DATASET_PATH, "train_set_average_importance"))


#   def analyzeBatch(batch):

#     #tensor with shape batch_size x 512 with each element being a token_id. 0 represents nothing
#     input_ids: Tensor = batch['input_ids'] 
#     #tensor with shape batch_size x 512 with each element indicating which input_ids should be attended to
#     # 1 should be attended to, 0 shouldn't be attended to
#     attention_mask: Tensor = batch['attention_mask'] 
#     #unsure
#     input_type_ids: Tensor = batch['token_type_ids']
#     tokens = torch.tensor([x for x in batch['input_ids'][0][1:] if x not in [tokenizer.sep_token_id, tokenizer.pad_token_id]])

#     print(tokens.size())
    
#   batch = next(iter(train_loader))
#   analyzeBatch(batch)
#   print("Average Importance", averageImportances[0].size())





# test()