import json, os, pickle, torch, logging, typing, numpy as np
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from tqdm import tqdm
from transformers import BertTokenizer, BertForMaskedLM, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import transformers as T
from logging.handlers import RotatingFileHandler
from src.utils.pickleUtils import pdump, pload, pjoin

from src.proecssing import correct_count
from transformers.models.bert.modeling_bert import SequenceClassifierOutput
from src.classes.datasets import DataItem, IMDBDataset
from torch import Tensor
import torch.linalg as lin
import functorch


def generateTiplets(
  dataset_name: str,
  batch_size: int,
  epoch_num: int,
  use_margin_loss: bool,
  lambda_weight: float,
  use_cache: bool,
  use_pinned_memory: bool
):
  batch_size = 1
  DATASET_NAME = dataset_name
  DATASET_PATH = f"datasets/{DATASET_NAME}/base"
  OUTPUT_PATH = f"checkpoints/{DATASET_NAME}/model"
  TRIPLETS_PATH = f"datasets/{DATASET_NAME}/augmented_triplets"
  
  #! convert these into arguments
  TOPK_NUM = 4
  DROPOUT_RATIO = 0.5
  MAX_MASKING_ATTEMPTS = 24

  import json


  sampling_ratio = 1

  tokenizer: T.BertTokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
  
  VOCAB_SIZE = tokenizer.vocab_size

  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

  train_set = pload(os.path.join(DATASET_PATH, "train_set"))
  train_texts = train_set["text"].tolist()[0:100]
  train_labels = train_set["label"].tolist()[0:100]

  train_encodings = tokenizer(train_texts, padding=True, truncation=True)
  train_dataset = IMDBDataset(labels=train_labels, encodings=train_encodings)

  train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=False,
    pin_memory=use_pinned_memory
    )
  
  num_classes = -1
  if len(train_loader.dataset[0]["labels"].shape) == 1:
    num_classes = train_loader.dataset[0]["labels"].shape[0]
  else:
    print("Invalid label shape")
    exit()






  importance_testing = open("./importance_testing", mode = 'w')



  def compute_importances(data_loader:DataLoader) -> list[Tensor]:
    model:BertForSequenceClassification = BertForSequenceClassification.from_pretrained(os.path.join(OUTPUT_PATH, 'best_epoch'), num_labels=num_classes) #type:ignore
    model.to(device)
    
    def get_gradient_norms(batch):
      input_ids:Tensor = batch['input_ids'].to(device)
      attention_mask:Tensor = batch['attention_mask'].to(device)
      labels:Tensor = batch['labels'].to(device)
      token_len = input_ids.shape[1]
      importances = []
      for i in range(input_ids.shape[0]):

        model.bert.embeddings.position_embeddings.weight.grad = None
        outputs:SequenceClassifierOutput | tuple[Tensor] = model.forward(input_ids=input_ids[i].unsqueeze(dim=0), attention_mask=attention_mask[i].unsqueeze(dim=0), labels=labels[i].unsqueeze(dim=0), return_dict=True)
        assert isinstance(outputs, SequenceClassifierOutput)

        loss = outputs['loss']
        loss.backward(retain_graph=True)
        torch.cuda.empty_cache()


        assert isinstance(model.bert.embeddings.position_embeddings.weight.grad, Tensor)
        assert isinstance(model.bert.embeddings.word_embeddings.weight.grad, Tensor)
        norm:Tensor = lin.norm(model.bert.embeddings.position_embeddings.weight.grad[0:token_len], ord=2, dim=1, dtype=torch.float64).detach().to(device)

        importances.append(norm)
      
      return importances

        


 


    all_importances: list = []
    for batch in tqdm(data_loader):
      importances = get_gradient_norms(batch)
      all_importances.append(importances)
    return all_importances

  def compute_average_importance(data_loader:DataLoader, all_importances) -> list[Tensor]:
    importance_sum = torch.zeros((VOCAB_SIZE,), dtype=torch.float64).to(device)
    importance_count = torch.zeros((VOCAB_SIZE,), dtype=torch.int32).to(device)
    total_count = 0
    for importances, batch in tqdm(zip(all_importances, data_loader)):
      for importance, item in zip(importances, batch['input_ids'].to(device)):
        m = max(item) + 1
        bincount = torch.bincount(item).to(device)
        importance_count[0:m] += bincount
        imp_bincount = item.bincount(importance)[0:m]*bincount[0:m]
        importance_sum[0:m] += imp_bincount
      total_count += batch['input_ids'].shape[0]
    averaged_importances = importance_sum/importance_count
    all_ave_token_importance = []
    for batch in tqdm(data_loader):
      ave_token_imp_batch = []
      m = batch['input_ids'].shape[0]
      for item in batch['input_ids'].to(device):
        ave_token_importance = torch.take(averaged_importances, item)
        ave_token_imp_batch.append(ave_token_importance)
      all_ave_token_importance.append(torch.stack(ave_token_imp_batch, dim=0))
    return all_ave_token_importance




  importance_path = os.path.join(DATASET_PATH, "importance")
  average_importance_path = os.path.join(DATASET_PATH, "average_importance")
  
  if os.path.exists(pjoin(importance_path)):
    sample_size, importance = pload(importance_path)
    if sample_size != len(train_texts):
      importance = compute_importances(train_loader)
      pdump((len(train_texts), importance), importance_path)
  else:
    importance = compute_importances(train_loader)
    pdump((len(train_texts),importance), importance_path)


  if os.path.exists(pjoin(average_importance_path)):
    sample_size, average_importance = pload(average_importance_path)
    if sample_size != len(train_texts):
      average_importance = compute_importances(train_loader)
      pdump((len(train_texts), average_importance), average_importance_path)
  else: 
    average_importance = compute_average_importance(train_loader, importance)
    pdump((len(train_texts), average_importance), average_importance_path)

  # exit()

  mlm_model: BertForMaskedLM = BertForMaskedLM.from_pretrained('bert-base-uncased') #type: ignore
  mlm_model: BertForMaskedLM = mlm_model.to(device) #type: ignore
  mlm_model.eval()

  def mask_data(data_loader:DataLoader, all_importances: list[Tensor], sampling_ratio, augment_ratio):
    print("herllo")
    triplets = []
    error_count = 0
    no_flip_count = 0
    no_flip_index = []
 
    token_idx_flip = []
    l = open("token_idx_flip", mode='w')

    for importances, batch in tqdm(zip(all_importances, data_loader)):

      label = []
      tokens = torch.tensor([x for x in batch['input_ids'][0][1:] if x not in [tokenizer.sep_token_id, tokenizer.pad_token_id]]) # could this be done better?
      # tokens = batch['input_ids']
      # assert tokens.size() == importances.size()
      
      orig_sample = tokenizer.decode(tokens,  clean_up_tokenization_spaces=True)
      # print(orig_sample)
      causal_mask, err_flag, maximum_score = mask_causal_words(tokens, batch, importances[0], topk=sampling_ratio)
      # exit()
      no_flip_index.append(err_flag)
      if err_flag:
        no_flip_count += 1
      else:
        token_idx_flip.append((np.argmax(causal_mask), len(causal_mask)))
      if 1 not in causal_mask:
        triplets.append((label, orig_sample, orig_sample, orig_sample, err_flag, maximum_score))
        continue


      for _ in range(augment_ratio):
        causal_masked_tokens = []
        noncausal_masked_tokens = []

        if sampling_ratio is None:
          causal_masked_tokens = [tokens[i] if causal_mask[i] == 0 else tokenizer.mask_token_id for i in range(len(tokens))]
          noncausal_masked_tokens = [tokens[i] if causal_mask[i] == 1 else tokenizer.mask_token_id for i in range(len(tokens))]

        elif type(sampling_ratio) == int:
          causal_indices = np.where(np.array(causal_mask) == 1)[0]
          noncausal_indices = np.where(np.array(causal_mask) == 0)[0]

          causal_mask_indices = np.random.choice(causal_indices, sampling_ratio)

          try:
            noncausal_mask_indices = np.random.choice(noncausal_indices, max(1, min(sampling_ratio, len(noncausal_indices))))
          except:
            noncausal_mask_indices = np.random.choice(causal_indices, sampling_ratio)
            error_count += 1

          causal_masked_tokens = [tokens[i] if i not in causal_mask_indices else tokenizer.mask_token_id for i in range(len(tokens))]
          noncausal_masked_tokens = [tokens[i] if i not in noncausal_mask_indices else tokenizer.mask_token_id for i in range(len(tokens))]
        else:
          pass

        causal_masked_sample = tokenizer.decode(causal_masked_tokens,  clean_up_tokenization_spaces=True)
        noncausal_masked_sample = tokenizer.decode(noncausal_masked_tokens, clean_up_tokenization_spaces=True)

        _, labels = torch.max(batch['labels'], dim=1)
    
        if labels[0] == 0: label = [0, 1]
        elif labels[0] == 1: label = [1, 0]
        triplets.append((label, orig_sample, causal_masked_sample, noncausal_masked_sample, err_flag, maximum_score))
    print(f"Error count: {error_count}")
    print(f"No flip count: {no_flip_count}")
    for i in token_idx_flip:
      l.write(str(i)+'\n')
      
    return triplets, no_flip_index




  json_file = open(f"mask_candidate_sample_{dataset_name}.json", mode='w')

  tokenizer: T.BertTokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
  model:BertForSequenceClassification = BertForSequenceClassification.from_pretrained(os.path.join(OUTPUT_PATH, 'best_epoch')) #type:ignore
  model.to(device)
  model.eval()
  def mask_causal_words(tokens:Tensor, batch:DataItem, importances: Tensor, topk=1):
    masking_attempts = len(tokens) if len(tokens)<=MAX_MASKING_ATTEMPTS else MAX_MASKING_ATTEMPTS
    importances = importances[0:len(tokens)]
    dropout = torch.nn.Dropout(0.5)
    causal_mask = [0 for _ in range(len(tokens))]
    all_importance_indices = torch.argsort(importances, descending=True)[0:masking_attempts]
    err_flag = False
    find_flag = False

    input_ids:Tensor = batch['input_ids'].squeeze().repeat((TOPK_NUM,)).reshape(TOPK_NUM, -1).to(device)
    attention_mask:Tensor = batch['attention_mask'].expand(TOPK_NUM, -1).to(device)
    token_type_ids:Tensor = batch['token_type_ids'].expand(TOPK_NUM, -1).to(device) #! is token_type_ids actually used anywhere???

    masked_input_ids = batch['input_ids'].squeeze().repeat((masking_attempts,)).reshape(masking_attempts, -1).to(device)
    masked_attention_mask = batch['attention_mask'].expand(masking_attempts, -1).to(device)
    masked_token_type_ids = batch['token_type_ids'].expand(masking_attempts, -1).to(device)

    # print(type(input_ids))
    # print(input_ids.shape)
    # print(masked_input_ids.shape)
    # print(masked_attention_mask.shape)
    # print(masked_token_type_ids.shape)
    # print(all_importance_indices.shape)



    fake_labels = torch.ones((masking_attempts, ))
    
    masked_train = IMDBDataset({
      'input_ids': masked_input_ids,
      'attention_mask': masked_attention_mask,
      'token_type_ids': masked_token_type_ids,
      'importance_indices': all_importance_indices,
    }, fake_labels)

    masked_train_loader = DataLoader(masked_train, batch_size=4, shuffle=False)
    logits = []

    for masked_batch in masked_train_loader:
      masked_input_ids: Tensor = masked_batch['input_ids'].to(device) # 4 x 313
      masked_attention_mask: Tensor = masked_batch['attention_mask'].to(device) # 4 x 313
      masked_token_type_ids: Tensor = masked_batch['token_type_ids'].to(device) # 4 x 313
      importance_indices: Tensor = masked_batch['importance_indices'].to(device)# 4
      masked_input_embeds: Tensor = mlm_model.bert.embeddings.word_embeddings(masked_input_ids) #4 x 313 x 768

      #dropout some of the embeddings at random
      for mi_i, topk_i in zip(range(masked_input_embeds.size(0)), importance_indices):
        masked_input_embeds[mi_i][topk_i + 1] = dropout(masked_input_embeds[mi_i][topk_i + 1])
      
      #get predicted words from mlm model given the partially missing embeds
      with torch.no_grad():
        outputs = mlm_model(attention_mask = masked_attention_mask, token_type_ids = masked_token_type_ids, inputs_embeds = masked_input_embeds)
        predictions = outputs[0] # 4 x 313 x 30522, just a casual 38 million numbers. shape(batch_size, sequence_length, config.vocab_size)
      
      #search through and find top k logits, in this case 4. 
      topk_logit_indices = torch.topk(predictions, TOPK_NUM, dim=-1)[1] # 4 sequences x 313 tokens x 4 


      

      #for the top k most important tokens, get their respective k candidates
      # mask_candidates = [topk_logits[importance_index + 1] for importance_index, topk_logits in zip(importance_indices, topk_logit_indices)]
      #! Switching to this solution resulted in a 15% increase in performance.
      def select_mc(topk_logits: Tensor, importance_idx: Tensor):
        return topk_logits.index_select(0, importance_idx).squeeze(0)
      batch_select_mc = functorch.vmap(select_mc)
      mask_candidates = batch_select_mc(topk_logit_indices, importance_indices.add(1).unsqueeze(1))


      for importance_index, mask_candidate in zip(importance_indices, mask_candidates):
        if importances[importance_index] == 0:
          continue
        recon_input_ids = input_ids.clone()
        for i, mc in enumerate(mask_candidate):
          recon_input_ids[i][importance_index + 1] = mc
        
        with torch.no_grad():
          #largest performance hog is this area right here. roughly 45% of time is spent in this model
          recon_outputs = model(recon_input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
          _, recon_prediction = torch.max(recon_outputs[0], dim=1)



        if len(torch.unique(recon_prediction)) != 1:
          j = {
              "original": tokenizer.decode(input_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True),
              "original_word": tokenizer.decode(torch.unsqueeze(input_ids[0][importance_index + 1], dim=0)),
              "replacement_words": [tokenizer.decode(mc) for mc in mask_candidate],
              "replacement_position": importance_index.item() + 1,
              "replacing_token_caused_flip": find_flag 
              }
          j.update(
            {
              f"candidate_{i}": tokenizer.decode(recon_input_ids[i],skip_special_tokens=True, clean_up_tokenization_spaces=True) for i in range(recon_input_ids.shape[0]) 
            }
          )
          json_file.write(json.dumps(j, indent=2))
          causal_mask[importance_index] = 1
          find_flag = True
          break

      if find_flag:
        break

    if 1 not in causal_mask:
      causal_mask[all_importance_indices[0]] = 1
      err_flag = True
      return causal_mask, err_flag, 0
    
    return causal_mask, err_flag, 0



  sampling_ratio = 1
  augment_ratio = 1

  triplets_train, no_flip_idx_train = mask_data(train_loader, average_importance, sampling_ratio=sampling_ratio, augment_ratio=augment_ratio)

  triplets_json:list[dict] = []
  triplets_pickle:dict[str,list] = {
    "labels": [],
    "anchor_texts": [],
    "positive_texts": [],
    "negative_texts": [],
    "triplet_sample_masks": []
  }
  for x in triplets_train:
    triplets_json.append(
      {
        "label": x[0],
        "anchor_text": x[1],
        "positive_text": x[3],
        "negative_text": x[2],
        "triplet_sample_mask": x[4]
      }
    )
    triplets_pickle["labels"].append(x[0])
    triplets_pickle["anchor_texts"].append(x[1])
    triplets_pickle["positive_texts"].append(x[3])
    triplets_pickle["negative_texts"].append(x[2])
    triplets_pickle["triplet_sample_masks"].append(x[4])


  if not os.path.exists(TRIPLETS_PATH):
    os.mkdir(TRIPLETS_PATH)

  pdump(triplets_pickle, os.path.join(TRIPLETS_PATH, "augmented_triplets"))
  with open(os.path.join(TRIPLETS_PATH, "augmented_triplets.json"), mode='w') as f:
    json.dump(triplets_json, f, indent=2)

