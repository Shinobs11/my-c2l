from dataset_splitting.splitYelpDatasets import split_dataset as split_yelp


def split_datasets(train_size: int = 300, test_size: int = 100, token_limit: int = 512):
  split_yelp(train_size=train_size, test_size=test_size, token_limit=token_limit)