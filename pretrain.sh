#!/bin/bash




# DATASET=fake_news_1
# DATASET=imdb
# DATASET=original_augmented_1x_aclImdb
DATASET=yelp_chi
echo "Starting preTraining script..."
python3 main.py \
--pretrain \
--dataset-name $DATASET \
--batch-size 8 \
--epoch-num 5

