#!/bin/bash
# DATASET=original_augmented_1x_aclImdb
DATASET=yelp_chi
echo "Starting contrastive learning script..."
for i in "0"
do
echo "Starting constrastive learning $0..."
python3 main.py \
--constrastive-train \
--dataset-name $DATASET \
--batch-size 2 \
--epoch-num 5
done
