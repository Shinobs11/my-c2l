#!/bin/bash





# DATASET=original_augmented_1x_aclImdb
DATASET=yelp_chi

python3 main.py \
--generate-triplets \
--dataset-name $DATASET \
--batch-size 1 \
--epoch-num 1 \
--use-margin-loss \
--use-cache \
--use-pinned-memory


# viztracer --ignore_c_function main.py \
# --generate-triplets \
# --dataset-name $DATASET \
# --batch-size 1 \
# --epoch-num 1 \
# --use-margin-loss \
# --use-cache \
# --use-pinned-memory

# py-spy record --rate 1000 --format speedscope -o profiles/generate-triplets-post-mc-fix.json -- python3 main.py \
# --generate-triplets \
# --dataset-name $DATASET \
# --batch-size 1 \
# --epoch-num 1 \
# --use-margin-loss \
# --use-cache \
# --use-pinned-memory
