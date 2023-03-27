# import json, os, pandas as pd
# from src.utils.pickleUtils import pload, pdump
# import ast

# def splitAugmentedSet(dataset_name:str):
#   base_path = f"datasets/{dataset_name}"
#   aug_base_path = f"{base_path}/augmented_triplets"
#   base_base_path = f"{base_path}/base"
#   aug_set_path = f"{aug_base_path}/augmented_triplets.json"
#   aug_set_file = open(aug_set_path, mode='r')
#   aug_set = json.load(aug_set_file)

#   aug_dict = {
#     "anchor_text": [],
#     "positive_text": [],
#     "negative_text": [],
#     "label": [],
#     "triplet_sample_mask": []
#   }
#   for x in aug_set:
#     for k, v in aug_dict.items():
#       v.append(x[k])
  
#   aug_df = pd.DataFrame.from_dict(aug_dict)
  
#   train_set = aug_df
#   valid_set = pd.read_csv(f"{base_base_path}/valid_set.csv")
#   test_set = pd.read_csv(f"{base_base_path}/test_set.csv") 

#   valid_set['label'] = [ast.literal_eval(x) for x in valid_set['label']]
#   test_set['label'] = [ast.literal_eval(x) for x in test_set['label']]
  
#   sets = ((train_set, "train_set"), (valid_set, "valid_set"), (test_set, "test_set"))
  
#   for set, set_name in sets:
#     set.to_csv(f"{aug_base_path}/{set_name}.csv", index=False)
#     pdump(set, f"{aug_base_path}/{set_name}")
#     print(f"Saved {set_name} to {aug_base_path}/{set_name}.csv")


# if __name__ == "__main__":
#   splitAugmentedSet("original_augmented_1x_aclImdb")
#   # splitAugmentedSet("imdb")
#   splitAugmentedSet("yelp_chi")



