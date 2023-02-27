#!/bin/bash




# DATASET=fake_news_1
DATASET=imdb
echo "Starting preTraining script..."
for i in "0"
do
echo "Starting preTraining $0..."
python3 main.py \
--evaluate \
--dataset-name $DATASET \
--batch-size 4 \
--epoch-num 1 \
--use-margin-loss \
--use-cache
done
