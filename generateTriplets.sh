#!/bin/bash





DATASET=original_augmented_1x_aclImdb
# DATASET=yelp_chi

PJRT_DEVICE=GPU python3 main.py \
--generate-triplets \
--dataset-name $DATASET \
--use-pinned-memory \
--sampling-ratio 4 \
--augment-ratio 3 \
--dropout-ratio 0.5 \
--topk-num 8 \
--max-masking-attempts 0 



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
