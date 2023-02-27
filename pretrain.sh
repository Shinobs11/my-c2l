#!/bin/bash




# DATASET=fake_news_1
# DATASET=imdb
DATASET=original_augmented_1x_aclImdb
# DATASET=yelp_chi
echo "Starting preTraining script..."
for i in "0"
do
echo "Starting preTraining $0..."
python3 main.py \
--pretrain \
--dataset-name $DATASET \
--batch-size 8 \
--epoch-num 10 \
--use-margin-loss \
--use-cache
done
