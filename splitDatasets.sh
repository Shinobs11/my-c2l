#!/bin/bash


python3 splitIMDBDatasets.py
python3 splitJSONIMDBDatasets.py --dataset-path datasets/original_augmented_1x_aclImdb/base
python3 splitYelpDatasets.py