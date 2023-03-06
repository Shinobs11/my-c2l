#!/bin/bash





DATASET=yelp_chi
echo "Evaluating base model"
python3 main.py \
--evaluate \
--dataset-name $DATASET \
--batch-size 4

sleep 1

python3 main.py \
--evaluate \
--dataset-name $DATASET \
--batch-size 4 \
--use-cl-model


